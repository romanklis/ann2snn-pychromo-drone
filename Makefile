# ANN2SNN 6-DoF drone in PyChrono
#
# Everything runs inside the PyChrono container. The working tree is bind-mounted
# at /work (where the editable install points), so these targets always execute
# the current checkout without rebuilding the image. Only `build` needs Docker
# build access; `clean` touches host files only.

IMAGE       ?= drone6dof:latest
TRAIN_IMAGE ?= drone6dof-train:latest
DASH_IMAGE  ?= drone6dof-dashboard:latest
CONTROLLER  ?= ds_guidance
CAMERA      ?= orbit
SECONDS     ?= 10
PORT        ?= 8080
COMPOSE     ?= docker-compose
XVFB        ?= bash scripts/xvfb.sh

# working-tree mount + output volume; X11 bits only for GUI targets
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)
MOUNT   = -v $(CURDIR):/work -w /work
OUTVOL  = -v $(CURDIR)/out:/data
X11     = --network host -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix:rw
RUN     = docker run --rm --user $(HOST_UID):$(HOST_GID) $(MOUNT)

.DEFAULT_GOAL := help

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## build the Docker image
	docker build -t $(IMAGE) .

shell: ## open a shell inside the container
	docker run --rm -it --user $(HOST_UID):$(HOST_GID) $(MOUNT) $(IMAGE) bash

demo: ## interactive Irrlicht demo (in the container)
	$(RUN) $(X11) $(IMAGE) python -m drone6dof --vis irrlicht \
		--controller $(CONTROLLER) --camera $(CAMERA)

pid: ## interactive Irrlicht demo with the PID baseline (collides with the pillar)
	$(RUN) $(X11) $(IMAGE) python -m drone6dof --vis irrlicht \
		--controller pid --camera $(CAMERA)

headless: ## headless run -> out/telemetry.csv + out/trajectory.npz (in the container)
	mkdir -p out
	$(RUN) $(OUTVOL) $(IMAGE) python -m drone6dof --vis none \
		--controller $(CONTROLLER) --seconds $(SECONDS) \
		--out /data/telemetry.csv --traj /data/trajectory.npz --json

frames: ## record PNG snapshots to out/frames (headless, in the container)
	mkdir -p out/frames
	$(RUN) $(OUTVOL) $(IMAGE) $(XVFB) python -m drone6dof --vis irrlicht \
		--controller $(CONTROLLER) --record-dir /data/frames --record-every 5

test: ## run the full test suite in the container (under Xvfb)
	$(RUN) $(IMAGE) $(XVFB) python -m pytest -q

test-fast: ## run tests in the container, skipping the slow parity test
	$(RUN) $(IMAGE) $(XVFB) python -m pytest -q -m "not slow"

compile: ## byte-compile the sources in the container
	$(RUN) $(IMAGE) python -m compileall -q src tests tools

train-image: ## build the torch training image
	docker build -f Dockerfile.train -t $(TRAIN_IMAGE) .

train: ## distil the connectome ANN/SNN and write weights/ (in the training image)
	mkdir -p weights
	$(RUN) $(TRAIN_IMAGE) python -m drone6dof.train \
		--out weights/quad6dof_connectome.npz \
		--ref-io weights/quad6dof_reference_io.npz

train-shell: ## shell in the training image
	docker run --rm -it --user $(HOST_UID):$(HOST_GID) $(MOUNT) $(TRAIN_IMAGE) bash

dashboard-build: ## build the dashboard image (node build + Flask)
	docker build -f Dockerfile.dashboard -t $(DASH_IMAGE) .

dashboard: ## build + run the comparison dashboard (http://localhost:$(PORT))
	docker build -f Dockerfile.dashboard -t $(DASH_IMAGE) .
	docker run --rm -p $(PORT):8080 $(DASH_IMAGE)

dashboard-shell: ## shell in the dashboard image
	docker run --rm -it --entrypoint sh $(DASH_IMAGE)

docker-headless: ## run the headless compose service (writes ./out/telemetry.csv)
	$(COMPOSE) run --rm headless

docker-interactive: ## run the interactive compose service (needs: xhost +local:docker)
	$(COMPOSE) run --rm interactive

docker-test: ## run the test compose service
	$(COMPOSE) run --rm test

clean: ## remove host build artifacts and outputs
	rm -rf out build dist .pytest_cache src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: help build shell demo pid headless frames test test-fast compile \
	train-image train train-shell dashboard-build dashboard dashboard-shell \
	docker-headless docker-interactive docker-test clean
