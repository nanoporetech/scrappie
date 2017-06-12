#include <assert.h>
#include <err.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fast5_interface.h"
#include "util.h"


event_table read_events(hid_t hdf5file, const char * tablepath, const float sample_rate);


struct _gop_data {
	const char * prefix;
	int latest;
};

typedef struct {
	//  Information for scaling raw data from ADC values to pA
	float digitisation;
	float offset;
	float range;
	float sample_rate;
} fast5_raw_scaling;

herr_t group_op_func (hid_t loc_id, const char *name, const H5L_info_t *info,
			void *operator_data){
	struct _gop_data * gop = (struct _gop_data *) operator_data;
	if(strncmp(name, gop->prefix, strlen(gop->prefix)) == 0){
		const size_t name_l = strlen(name);
		int analysis_number = atoi(name + name_l - 3);
		if(analysis_number > gop->latest){
			gop->latest = analysis_number;
		}
	}

	return 0;
}

int get_latest_group(hid_t file, const char * root, const char * prefix){
	assert(NULL != root);
	assert(NULL != prefix);

	struct _gop_data gop_data = {prefix, -1};
        hid_t grp = H5Gopen(file, root, H5P_DEFAULT);
	if(grp < 0){
		warnx("Failed to open group '%s' at %s:%d.", root, __FILE__, __LINE__);
		return gop_data.latest;
	}

	herr_t status = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, group_op_func, &gop_data);
	if(status < 0 || gop_data.latest < 0){
		warnx("Error trying to find group of form '%s/%s_XXX'.", root, prefix);
	}
	H5Gclose(grp);

	return gop_data.latest;
}


range_t get_segmentation(hid_t file, int analysis_no, const char * segloc1, const char * segloc2){
	assert(NULL != segloc1);
	assert(NULL != segloc2);
	range_t segcoord = {-1, -1};


	if(analysis_no < 0){
		analysis_no = get_latest_group(file, "/Analyses", segloc1);
		if(analysis_no < 0){
			warnx("No segmentation group found.  You may want to specify it using the --segmentation option.\n"
			      "Will use all events for calling.  See READ.md for details.");
			return segcoord;
		}
	}

	int segnamelen = strlen(segloc1) + strlen(segloc2) + 24;
	char * segname = calloc(segnamelen, sizeof(char));
	(void)snprintf(segname, segnamelen, "/Analyses/%s_%03d/Summary/%s", segloc1, analysis_no, segloc2);
	hid_t seggroup = H5Gopen(file, segname, H5P_DEFAULT);
	if(seggroup < 0){
		warnx("Failed to open group '%s' to read segmentation from.  Will use all events.", segname);
		goto clean1;
	}

	//  Get start location
	hid_t temp_start = H5Aopen_name(seggroup, "first_sample_template");
	if(temp_start < 0){
		// Try old-style location
		temp_start = H5Aopen_name(seggroup, "start_index_temp");
	}
	if(temp_start >= 0){
		H5Aread(temp_start, H5T_NATIVE_INT, &segcoord.start);
		H5Aclose(temp_start);
	} else {
		warnx("Segmentation group '%s_%03d/Summary/%s'does not contain valid attribute.",
			segloc1, analysis_no, segloc2);
	}

	// Get end location
	hid_t temp_end = H5Aopen_name(seggroup, "duration_template");
	if(temp_end >= 0){
		H5Aread(temp_end, H5T_NATIVE_INT, &segcoord.end);
		H5Aclose(temp_end);
		segcoord.end += segcoord.start;
	} else {
		// Try old style coordinates
		temp_end = H5Aopen_name(seggroup, "end_index_temp");
		if(temp_end >= 0){
			H5Aread(temp_end, H5T_NATIVE_INT, &segcoord.end);
			H5Aclose(temp_end);
			// Use Python-like convention where final index is exclusive upper bound
			segcoord.end += 1;
		} else {
			warnx("Segmentation group '%s_%03d/Summary/%s' does not contain valid attribute.",
				segloc1, analysis_no, segloc2);
		}
	}


	H5Gclose(seggroup);
clean1:
	free(segname);

	return segcoord;
}


event_table read_events(hid_t hdf5file, const char * tablepath, const float sample_rate){
	assert(NULL != tablepath);
	event_table ev = {0, 0, 0, NULL};

	hid_t dset = H5Dopen(hdf5file, tablepath, H5P_DEFAULT);
	if(dset < 0){
		warnx("Failed to open dataset '%s' to read events from.", tablepath);
		return ev;
	}

	hid_t space = H5Dget_space(dset);
	if(space < 0){
		warnx("Failed to create copy of dataspace for event table %s.", tablepath);
		goto clean1;
	}

	hsize_t dims[1];
	H5Sget_simple_extent_dims (space, dims, NULL);
	const size_t nevent = dims[0];
	hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(event_t));
	if(memtype < 0){
		warnx("Failed to create memory representation for event table %s:%d.", __FILE__, __LINE__);
		goto clean2;
	}
	//  Using doubles to store time and length is a poor choice but forced by the
	//  need to be compatible with both Albacore and Minknow files.
	H5Tinsert(memtype, "start", HOFFSET(event_t, start), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "length", HOFFSET(event_t, length), H5T_NATIVE_FLOAT);
	H5Tinsert(memtype, "mean", HOFFSET(event_t, mean), H5T_NATIVE_FLOAT);
	H5Tinsert(memtype, "stdv", HOFFSET(event_t, stdv), H5T_NATIVE_FLOAT);
	event_t * events = calloc(nevent, sizeof(event_t));
	if(NULL == events){
		warnx("Failed to allocate memory for events");
		goto clean3;
	}
	herr_t status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, events);
	if(status < 0){
		free(events);
		warnx("Failed to read events out of dataset %s.", tablepath);
		goto clean3;
	}

	for(int ev=0 ; ev < nevent ; ev++){
		// Convert to samples if necessary
		events[ev].start = round(events[ev].start * sample_rate);
		events[ev].length = roundf(events[ev].length * sample_rate);
	}

	for(int ev=0 ; ev < nevent ; ev++){
		// Negative means unassigned
		events[ev].pos = -1;
		events[ev].state = -1;
	}
	ev = (event_table){nevent, 0, nevent, events};


clean3:
	H5Tclose(memtype);
clean2:
	H5Sclose(space);
clean1:
	H5Dclose(dset);

	return ev;
}


event_table read_detected_events(const char * filename, int analysis_no, const char * segloc1, const char * segloc2, int seganalysis_no){
	assert(NULL != filename);
	assert(NULL != segloc1);
	assert(NULL != segloc2);
	event_table ev = {0, 0, 0, NULL};
	const size_t rootstr_len = 36;

	hid_t hdf5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (hdf5file < 0) {
		warnx("Failed to open %s for reading.", filename);
		return ev;
	}
	H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

	if(analysis_no < 0){
		analysis_no = get_latest_group(hdf5file, "/Analyses", "EventDetection");
		if(analysis_no < 0){ return ev; }
	}

	//  Find group name of read (take first if there are multiple)
	char * root = calloc(rootstr_len, sizeof(char));
	(void)snprintf(root, rootstr_len, "/Analyses/EventDetection_%03d/Reads/", analysis_no);
	ssize_t size = H5Lget_name_by_idx(hdf5file, root, H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 0, H5P_DEFAULT);
	if(size < 0){
		warnx("Failed find read name under %s.", root);
		goto cleanup1;
	}
	char * name = calloc(1 + size, sizeof(char));
	H5Lget_name_by_idx(hdf5file, root, H5_INDEX_NAME, H5_ITER_INC, 0, name, 1 + size, H5P_DEFAULT);

	//  Prepare event group
	char * event_group = calloc(rootstr_len + size + 7, sizeof(char));
	(void)snprintf(event_group, rootstr_len + size + 7, "%s%s/Events", root, name);
        free(name);
	ev = read_events(hdf5file, event_group, 1.0);
	free(event_group);

	//  Add segmentation information
	range_t  segcoord = get_segmentation(hdf5file, seganalysis_no, segloc1, segloc2);
        ev.start = (segcoord.start >= 0) ? segcoord.start : 0;
	ev.end = (segcoord.end >= 0 && segcoord.end <= ev.end) ? segcoord.end : ev.end;

cleanup1:
	free(root);
	H5Fclose(hdf5file);

	return ev;
}


float read_float_attribute(hid_t group, const char * attribute){
	float val = NAN;
	if(group < 0){
		warnx("Invalid group passed to %s:%d.", __FILE__, __LINE__);
		return val;
	}

	hid_t attr = H5Aopen(group, attribute, H5P_DEFAULT);
	if(attr < 0){
		warnx("Failed to open attribute '%s' for reading.", attribute);
		return val;
	}

	H5Aread(attr, H5T_NATIVE_FLOAT, &val);
	H5Aclose(attr);

	return val;
}


fast5_raw_scaling get_raw_scaling(hid_t hdf5file){
	// Add 1e-5 to sensible sample rate as a sentinel value
	fast5_raw_scaling scaling = {NAN, NAN, NAN, NAN};
	const char * scaling_path = "/UniqueGlobalKey/channel_id";

	hid_t scaling_group = H5Gopen(hdf5file, scaling_path, H5P_DEFAULT);
	if(scaling_group < 0){
		warnx("Failed to group %s.", scaling_path);
		return scaling;
	}

	scaling.digitisation = read_float_attribute(scaling_group, "digitisation");
	scaling.offset = read_float_attribute(scaling_group, "offset");
	scaling.range = read_float_attribute(scaling_group, "range");
	scaling.sample_rate = read_float_attribute(scaling_group, "sampling_rate");

	H5Gclose(scaling_group);

	return scaling;
}


raw_table read_raw(const char * filename, bool scale_to_pA){
	assert(NULL != filename);
	raw_table rawtbl = {0, 0, 0, NULL};

	hid_t hdf5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(hdf5file < 0){
		warnx("Failed to open %s for reading.", filename);
		return rawtbl;
	}
	H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

	const char * root = "/Raw/Reads/";
	const int rootstr_len = strlen(root);
	ssize_t size = H5Lget_name_by_idx(hdf5file, root, H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 0, H5P_DEFAULT);
	if(size < 0){
		warnx("Failed find read name under %s.", root);
		goto cleanup1;
	}
	char * name = calloc(1 + size, sizeof(char));
	H5Lget_name_by_idx(hdf5file, root, H5_INDEX_NAME, H5_ITER_INC, 0, name, 1 + size, H5P_DEFAULT);

	// Create group name
	char * signal_path = calloc(rootstr_len + size + 8, sizeof(char));
	(void)snprintf(signal_path, rootstr_len + size + 8, "%s%s/Signal", root, name);
	free(name);


	hid_t dset = H5Dopen(hdf5file, signal_path, H5P_DEFAULT);
	if(dset < 0){
		warnx("Failed to open dataset '%s' to read raw signal from.", signal_path);
		goto cleanup2;
	}

	hid_t space = H5Dget_space(dset);
	if(space < 0){
		warnx("Failed to create copy of dataspace for raw signal %s.", signal_path);
		goto cleanup3;
	}
	hsize_t nsample;
	H5Sget_simple_extent_dims (space, &nsample, NULL);
	float * rawptr = calloc(nsample, sizeof(float));
	herr_t status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawptr);
	if(status < 0){
		free(rawptr);
		warnx("Failed to read raw data from dataset %s.", signal_path);
		goto cleanup4;
	}
	rawtbl = (raw_table){nsample, 0, nsample, rawptr};

	if(scale_to_pA){
		const fast5_raw_scaling scaling = get_raw_scaling(hdf5file);
		const float raw_unit = scaling.range / scaling.digitisation;
		for(size_t i=0 ; i < nsample ; i++){
			rawptr[i] = (rawptr[i] + scaling.offset) * raw_unit;
		}
	}


cleanup4:
	H5Sclose(space);
cleanup3:
	H5Dclose(dset);
cleanup2:
	free(signal_path);
cleanup1:
	H5Fclose(hdf5file);

	return rawtbl;
}


event_table read_albacore_events(const char * filename, int analysis_no, const char * section){
	assert(NULL != filename);
	assert(NULL != section);
	event_table ev = {0, 0, 0, NULL};

	hid_t hdf5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(hdf5file < 0){
		warnx("Failed to open %s for reading.", filename);
		return ev;
	}
	H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

	if(analysis_no < 0){
		analysis_no = get_latest_group(hdf5file, "/Analyses", "Basecall_1D_");
		if(analysis_no < 0){ return ev; }
	}

	const int loclen = 45 + strlen(section);
	char * event_group = calloc(loclen, sizeof(char));
	(void)snprintf(event_group, loclen, "/Analyses/Basecall_1D_%03d/BaseCalled_%s/Events", analysis_no, section);

	// Read sample rate from attribute in file
	const fast5_raw_scaling scaling = get_raw_scaling(hdf5file);

	ev = read_events(hdf5file, event_group, scaling.sample_rate);
	free(event_group);
	H5Fclose(hdf5file);

	return ev;
}


void write_annotated_events(hid_t hdf5file, const char * readname, const event_table ev, hsize_t chunk_size, int compression_level){
	assert(compression_level >= 0 && compression_level <= 9);
	// Memory representation
	hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(event_t));
	if(memtype < 0){
		warnx("Failed to create memroy representation for event table %s:%d.", __FILE__, __LINE__);
		goto clean1;
	}
	H5Tinsert(memtype, "start", HOFFSET(event_t, start), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "length", HOFFSET(event_t, length), H5T_NATIVE_FLOAT);
	H5Tinsert(memtype, "mean", HOFFSET(event_t, mean), H5T_NATIVE_FLOAT);
	H5Tinsert(memtype, "stdv", HOFFSET(event_t, stdv), H5T_NATIVE_FLOAT);
	H5Tinsert(memtype, "pos", HOFFSET(event_t, pos), H5T_NATIVE_INT);

	// File representation
	hid_t filetype = H5Tcreate(H5T_COMPOUND, 4 * 5);
	if(filetype < 0){
		warnx("Failed to create file representation for event table %s:%d.", __FILE__, __LINE__);
		goto clean2;
	}

	H5Tinsert(filetype, "start", 0, H5T_STD_U32LE);
	H5Tinsert(filetype, "length", 4, H5T_STD_U32LE);
	H5Tinsert(filetype, "mean", 4 * 2, H5T_IEEE_F32LE);
	H5Tinsert(filetype, "stdv", 4 * 3, H5T_IEEE_F32LE);
	H5Tinsert(filetype, "pos", 4 * 4, H5T_STD_I32LE);

	// Create dataset
	const hsize_t dims = ev.n;
	hid_t space = H5Screate_simple(1, &dims, NULL);
	if(space < 0){
		warnx("Failed to allocate dataspace for event table %s:%d.", __FILE__, __LINE__);
		goto clean3;
	}
	// Enable compression if available
	hid_t properties = H5P_DEFAULT;
	if(compression_level > 0){
		properties = H5Pcreate(H5P_DATASET_CREATE);
		if(properties < 0){
			warnx("Failed to create properties structure to write out compressed data structure.\n");
			properties = H5P_DEFAULT;
		} else {
			H5Pset_shuffle(properties);
			H5Pset_deflate(properties, compression_level);
			H5Pset_chunk(properties, 1, &chunk_size);
		}
	}

	hid_t dset = H5Dcreate(hdf5file, readname, filetype, space, H5P_DEFAULT, properties, H5P_DEFAULT);
	if(dset < 0){
		warnx("Failed to create dataset for event table %s:%d.", __FILE__, __LINE__);
		goto clean4;
	}


	// Write data
	herr_t writeret = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, ev.event);
	if(writeret < 0){
		warnx("Failed to write dataset for event table %s:%d.", __FILE__, __LINE__);
	}

clean4:
	if(H5P_DEFAULT != properties){
		H5Pclose(properties);
	}
	H5Dclose(dset);
clean3:
	H5Sclose(space);
clean2:
	H5Tclose(filetype);
clean1:
	H5Tclose(memtype);
}


void write_annotated_raw(hid_t hdf5file, const char * readname, const raw_table rt, hsize_t chunk_size, int compression_level){
	return;
}


/**  Simple segmentation of a raw read by thresholding the MAD
 *
 *  The MAD of the raw signal is calculated for non-overlapping chunks and then
 *  thresholded to find regions at the beginning and end of the signal that have
 *  unusually low variation (generally a stall or open pore).  The threshhold is
 *  derived from the distribution of the calaculated MADs.
 *
 *  The threshold is chosen to be high since a single chunk above it will trigger
 *  the end of the trimming: the threshhold is chosen so it is unlikely to be
 *  exceeded in the leader but commonly exceeded in the main read.
 *
 *  @param rt Structure containing raw signal
 *  @param chunk_size Size of non-overlapping chunks
 *  @param perc  The quantile to be calculated to use for threshholding
 *
 *  @return A range structure containing new start and end for read
 **/
range_t trim_raw_by_mad(const raw_table rt, int chunk_size, float perc){
	assert(chunk_size > 1);
	assert(perc >= 0.0 && perc <= 1.0);
	const size_t nsample = rt.end - rt.start;
	const size_t nchunk = nsample / chunk_size;
	range_t range = {rt.start, rt.end};

	float * madarr = malloc(nchunk * sizeof(float));
	for(size_t i=0 ; i < nchunk ; i++){
		madarr[i] = madf(rt.raw + rt.start + i * chunk_size, chunk_size, NULL);
	}
	quantilef(madarr, nchunk, &perc, 1);

	const float thresh = perc;
	for(size_t i=0 ; i < nchunk ; i ++){
		if(madarr[i] > thresh){
			break;
		}
		range.start += chunk_size;
	}
	for(size_t i=nchunk ; i > 0 ; i--){
		if(madarr[i - 1] > thresh){
			break;
		}
		range.end -= chunk_size;
	}
	assert(range.start < rt.end);
	assert(range.end > rt.start);
	assert(range.end > range.start);

	free(madarr);

	return range;
}
