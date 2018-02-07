
centos_images := $(addprefix centos/, centos_7_0_1406 centos_7_1_1503 centos_7_2_1511 centos_7_3_1611 centos_7_4_1708)
opensuse_images := $(addprefix opensuse/, opensuse_42_3)
ubuntu_images := $(addprefix ubuntu/, ubuntu_14_04 ubuntu_16_04 ubuntu_17_10 ubuntu_18_04)

.PHONY:all
all: centos opensuse ubuntu

.PHONY:ubuntu
ubuntu: $(addsuffix /docker.log, ${ubuntu_images})


.PHONY:centos
centos: $(addsuffix /docker.log, ${centos_images})

.PHONY:centos
opensuse: $(addsuffix /docker.log, ${opensuse_images})


%/docker.log : %/Dockerfile
	docker build --tag scrappie:$(notdir $(patsubst %/, %, $(dir $@))) $(patsubst %/, %, $(dir $@)) > $@.tmp && mv $@.tmp $@
