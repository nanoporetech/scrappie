#include <assert.h>
#include <err.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "events.h"

event_table read_events(hid_t hdf5file, const char * tablepath, const float sample_rate);


struct _range { int start, end;};
struct _gop_data {
	const char * prefix;
	int latest;
};

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


struct _range get_segmentation(hid_t file, int analysis_no, const char * segmentation){
	assert(NULL != segmentation);
	struct _range segcoord = {-1, -1};

	if(analysis_no < 0){
		analysis_no = get_latest_group(file, "/Analyses", segmentation);
		if(analysis_no < 0){ return segcoord; }
	}

        int segnamelen = strlen(segmentation) + 37;
	char * segname = calloc(segnamelen, sizeof(char));
	(void)snprintf(segname, segnamelen, "/Analyses/%s_%03d/Summary/split_hairpin", segmentation, analysis_no);
	hid_t segloc = H5Gopen(file, segname, H5P_DEFAULT);
	if(segloc < 0){
		warnx("Failed to open group '%s' to read segmentation from.", segname);
		goto clean1;
	}

	hid_t temp_start = H5Aopen_name(segloc, "start_index_temp");
	if(temp_start > 0){
		H5Aread(temp_start, H5T_NATIVE_INT, &segcoord.start);
		H5Aclose(temp_start);
	} else {
		warnx("Segmentation group '%s'does not contain 'start_index_temp' attribute.", segmentation);
	}

	hid_t temp_end = H5Aopen_name(segloc, "end_index_temp");
	if(temp_end > 0){
		H5Aread(temp_end, H5T_NATIVE_INT, &segcoord.end);
		H5Aclose(temp_end);
		// Use Python-like convention where final index is exclusive upper bound
		segcoord.end += 1;
	} else {
		warnx("Segmentation group '%s' does not contain 'end_index_temp' attribute.", segmentation);
	}


	H5Gclose(segloc);
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


event_table read_detected_events(const char * filename, int analysis_no, const char * segmentation, int seganalysis_no){
	assert(NULL != filename);
	assert(NULL != segmentation);
	event_table ev = {0, 0, 0, NULL};
	const size_t rootstr_len = 36;

	hid_t hdf5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (hdf5file < 0) {
		warnx("Failed to open %s for reading.", filename);
		return ev;
	}

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
	struct _range  segcoord = get_segmentation(hdf5file, seganalysis_no, segmentation);
        ev.start = (segcoord.start >= 0) ? segcoord.start : 0;
	ev.end = (segcoord.end >= 0 && segcoord.end <= ev.end) ? segcoord.end : ev.end;

cleanup1:
	free(root);
	H5Fclose(hdf5file);

	return ev;
}


float albacore_sample_rate(hid_t hdf5file){
	// Add 1e-5 to sensible sample rate as a sentinel value
	float sample_rate = 4000.0 + 1e-5;
	const char * sample_rate_group = "/UniqueGlobalKey/channel_id";

	hid_t sample_group = H5Gopen(hdf5file, sample_rate_group, H5P_DEFAULT);
	if(sample_group < 0){
		warnx("Failed to group %s.", sample_rate_group);
		return sample_rate;
	}

	hid_t sample_attr = H5Aopen(sample_group, "sampling_rate", H5P_DEFAULT);
	if(sample_attr < 0){
		warnx("Failed to read sampling_rate attribute.");
		goto cleanup1;
	}

	H5Aread(sample_attr, H5T_NATIVE_FLOAT, &sample_rate);


	H5Aclose(sample_attr);
cleanup1:
	H5Gclose(sample_group);

	return sample_rate;
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

	if(analysis_no < 0){
		analysis_no = get_latest_group(hdf5file, "/Analyses", "Basecall_1D_");
		if(analysis_no < 0){ return ev; }
	}

	const int loclen = 45 + strlen(section);
	char * event_group = calloc(loclen, sizeof(char));
	(void)snprintf(event_group, loclen, "/Analyses/Basecall_1D_%03d/BaseCalled_%s/Events", analysis_no, section);

	// Read sample rate from attribute in file
	const float sample_rate = albacore_sample_rate(hdf5file);
	printf("Sample rate is %f\n", sample_rate);


	ev = read_events(hdf5file, event_group, sample_rate);
	free(event_group);
	H5Fclose(hdf5file);

	return ev;
}


void write_annotated_events(hid_t hdf5file, const char * readname, const event_table ev, hsize_t chunk_size, int compression_level){
	assert(compression_level >= 0 && compresion_level <= 9);
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
	hid_t filetype = H5Tcreate(H5T_COMPOUND, 8 + 4 * 4);
	if(filetype < 0){
		warnx("Failed to create file representation for event table %s:%d.", __FILE__, __LINE__);
		goto clean2;
	}

	H5Tinsert(filetype, "start", 0, H5T_STD_U32LE);
	H5Tinsert(filetype, "length", 4, H5T_STD_U32LE);
	H5Tinsert(filetype, "mean", 4 * 2, H5T_IEEE_F32LE);
	H5Tinsert(filetype, "stdv", 4 * 3, H5T_NATIVE_FLOAT);
	H5Tinsert(filetype, "pos", 4 * 4, H5T_NATIVE_INT);

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
