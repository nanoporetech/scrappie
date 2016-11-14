#define _GNU_SOURCE
#include <assert.h>
#include <hdf5.h>
#include <stdlib.h>
#include "read_events.h"


event_table read_events(const char * filename, const char * tablepath){
	assert(NULL != filename);
	assert(NULL != tablepath);
	hsize_t dims[1];


	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	hid_t dset = H5Dopen(file, tablepath, H5P_DEFAULT);

	hid_t space = H5Dget_space(dset);
	H5Sget_simple_extent_dims (space, dims, NULL);
	size_t nevent = dims[0];
	event_t * events = malloc(nevent * sizeof(event_t));

	hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(event_t));
	H5Tinsert(memtype, "start", HOFFSET(event_t, start), H5T_NATIVE_INT);
	H5Tinsert(memtype, "length", HOFFSET(event_t, length), H5T_NATIVE_INT);
	H5Tinsert(memtype, "mean", HOFFSET(event_t, mean), H5T_NATIVE_DOUBLE);
	H5Tinsert(memtype, "stdv", HOFFSET(event_t, stdv), H5T_NATIVE_DOUBLE);
	herr_t status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, events);

	//H5Dvlen_reclaim (memtype, space, H5P_DEFAULT, events);
	H5Tclose(memtype);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);

	return (event_table){nevent, events};
}


event_table read_detected_events(const char * filename, int analysis_no){
	assert(NULL != filename);

	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	char * root = NULL;
	(void)asprintf(&root, "/Analyses/EventDetection_%03d/Reads/", analysis_no);

	size_t size = 1 + H5Lget_name_by_idx(file, root, H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 0, H5P_DEFAULT);
	char * name = calloc(size, sizeof(char));
	H5Lget_name_by_idx(file, root, H5_INDEX_NAME, H5_ITER_INC, 0, name, size, H5P_DEFAULT);
	char * event_group = NULL;
	(void)asprintf(&event_group, "%s%s/Events", root, name);

        free(name);
	free(root);
	H5Fclose(file);

	event_table ev = read_events(filename, event_group);
	free(event_group);
	return ev;
}
