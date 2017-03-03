#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "events.h"


struct _pi { int x1, x2;};

struct _pi get_segmentation(hid_t file, int analysis_no, const char * segmentation){
	assert(NULL != segmentation);
	int start=0, end=0;

        int segnamelen = strlen(segmentation) + 37;
	char * segname = calloc(segnamelen, sizeof(char));
	(void)snprintf(segname, segnamelen, "/Analyses/%s_%03d/Summary/split_hairpin", segmentation, analysis_no);
	hid_t segloc = H5Gopen(file, segname, H5P_DEFAULT);

	hid_t temp_start = H5Aopen_name(segloc, "start_index_temp");
	H5Aread(temp_start, H5T_NATIVE_INT, &start);
	H5Aclose(temp_start);

	hid_t temp_end = H5Aopen_name(segloc, "end_index_temp");
	H5Aread(temp_end, H5T_NATIVE_INT, &end);
	H5Aclose(temp_end);


	H5Gclose(segloc);
	free(segname);
	return (struct _pi){start, end};
}


event_table read_events(const char * filename, const char * tablepath){
	assert(NULL != filename);
	assert(NULL != tablepath);
	hsize_t dims[1];


	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file < 0){ return (event_table){0, 0, 0, NULL};}

	hid_t dset = H5Dopen(file, tablepath, H5P_DEFAULT);

	hid_t space = H5Dget_space(dset);
	H5Sget_simple_extent_dims (space, dims, NULL);
	size_t nevent = dims[0];
	event_t * events = calloc(nevent, sizeof(event_t));

	hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(event_t));
	H5Tinsert(memtype, "start", HOFFSET(event_t, start), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "length", HOFFSET(event_t, length), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "mean", HOFFSET(event_t, mean), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "stdv", HOFFSET(event_t, stdv), H5T_NATIVE_DOUBLE);
	herr_t status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, events);

	for(int ev=0 ; ev < nevent ; ev++){
		// Negative means unassigned
		events[ev].pos = -1;
		events[ev].state = -1;
	}

	H5Tclose(memtype);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);

	return (event_table){nevent, 0, nevent, events};
}


event_table read_detected_events(const char * filename, int analysis_no, const char * segmentation, int seganalysis_no){
	assert(NULL != filename);

	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file < 0){ return (event_table){0, 0, 0, NULL};}

	char * root = calloc(36, sizeof(char));
	(void)snprintf(root, 36, "/Analyses/EventDetection_%03d/Reads/", analysis_no);

	size_t size = 1 + H5Lget_name_by_idx(file, root, H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 0, H5P_DEFAULT);
	char * name = calloc(size, sizeof(char));
	H5Lget_name_by_idx(file, root, H5_INDEX_NAME, H5_ITER_INC, 0, name, size, H5P_DEFAULT);
	char * event_group = calloc(36 + size + 7 - 1, sizeof(char));
	(void)snprintf(event_group, 36 + size + 7 - 1, "%s%s/Events", root, name);
        free(name);

	event_table ev = read_events(filename, event_group);
	free(event_group);
	free(root);


	struct _pi  index = get_segmentation(file, seganalysis_no, segmentation);
        ev.start = index.x1;
	ev.end = index.x2;
	H5Fclose(file);

	return ev;
}


event_table read_albacore_events(const char * filename, int analysis_no, const char * section){
	assert(NULL != filename);

	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file < 0){ return (event_table){0, 0, 0, NULL};}

	const int loclen = 45 + strlen(section);
	char * event_group = calloc(loclen, sizeof(char));
	(void)snprintf(event_group, loclen, "/Analyses/Basecall_1D_%03d/BaseCalled_%s/Events", analysis_no, section);

	event_table ev = read_events(filename, event_group);
	free(event_group);
	H5Fclose(file);

	return ev;
}


void write_annotated_events(hid_t hdf5file, const char * readname, const event_table ev){
	// Memory representation
	hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(event_t));
	H5Tinsert(memtype, "start", HOFFSET(event_t, start), H5T_NATIVE_INT);
	H5Tinsert(memtype, "length", HOFFSET(event_t, length), H5T_NATIVE_INT);
	H5Tinsert(memtype, "mean", HOFFSET(event_t, mean), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "stdv", HOFFSET(event_t, stdv), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "pos", HOFFSET(event_t, pos), H5T_NATIVE_INT);
	H5Tinsert(memtype, "state", HOFFSET(event_t, state), H5T_NATIVE_INT);
	// File representation
	hid_t filetype = H5Tcreate(H5T_COMPOUND, 8 * 6);
	H5Tinsert(filetype, "start", 8 * 0, H5T_STD_I64LE);
	H5Tinsert(filetype, "length", 8 * 1, H5T_STD_I64LE);
	H5Tinsert(filetype, "mean", 8 * 2, H5T_IEEE_F64LE);
	H5Tinsert(filetype, "stdv", 8 * 3, H5T_IEEE_F64LE);
	H5Tinsert(filetype, "pos", 8 * 4, H5T_STD_I64LE);
	H5Tinsert(filetype, "state", 8 * 5, H5T_STD_I64LE);
	// Create dataset
	hsize_t dims[1] = {ev.n};
	hid_t space = H5Screate_simple(1, dims, NULL);
	hid_t dset = H5Dcreate(hdf5file, readname, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Write data
	H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, ev.event);

	H5Dclose(dset);
	H5Sclose(space);
	H5Tclose(filetype);
	H5Tclose(memtype);
}
