
centos_images := $(addprefix centos/, centos_7_0_1406 centos_7_1_1503 centos_7_2_1511 centos_7_3_1611 centos_7_4_1708)
#  Debian Wheezy not supported due to cmake being too old
debian_images := $(addprefix debian/, jessie stretch)
opensuse_images := $(addprefix opensuse/, opensuse_42_3 opensuse_42_2)
ubuntu_images := $(addprefix ubuntu/, ubuntu_14_04 ubuntu_16_04 ubuntu_17_10 ubuntu_18_04)

.PHONY:all
all: centos debian opensuse ubuntu


.PHONY:centos
centos: $(addsuffix /docker.log, ${centos_images})

.PHONY:debian
debian: $(addsuffix /docker.log, ${debian_images})

.PHONY:opensuse
opensuse: $(addsuffix /docker.log, ${opensuse_images})

.PHONY:ubuntu
ubuntu: $(addsuffix /docker.log, ${ubuntu_images})


%/docker.log : %/Dockerfile
	docker build --tag scrappie:$(notdir $(patsubst %/, %, $(dir $@))) $(patsubst %/, %, $(dir $@)) > $@.tmp && mv $@.tmp $@
