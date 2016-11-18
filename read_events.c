#define _GNU_SOURCE
#include <assert.h>
#include <hdf5.h>
#include <stdlib.h>
#include "read_events.h"


struct _pi get_segmentation(hid_t file, int analysis_no){
	int start=0, end=0; 

	char * segname = NULL;
	(void)asprintf(&segname, "/Analyses/Segment_Linear_%03d/Summary/split_hairpin", analysis_no);
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

event_table read_events(const char * filename, const char * tablepath, struct _pi index){
	assert(NULL != filename);
	assert(NULL != tablepath);
	hsize_t dims[1];


	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file < 0){ return (event_table){0, 0, 0, NULL};}

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

	H5Dvlen_reclaim (memtype, space, H5P_DEFAULT, events);
	H5Tclose(memtype);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);

	return (event_table){nevent, index.x1, index.x2, events};
}


event_table read_detected_events(const char * filename, int analysis_no){
	assert(NULL != filename);

	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file < 0){ return (event_table){0, 0, 0, NULL};}
	struct _pi  index = get_segmentation(file, analysis_no);

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

	event_table ev = read_events(filename, event_group, index);
	free(event_group);
	return ev;
}
