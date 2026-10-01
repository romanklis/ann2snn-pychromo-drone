# PyChrono with the Irrlicht visualization module, plus ROS 2 Humble.
#
# No conda-forge / official-channel PyChrono build includes a renderer or
# sensors (conda-forge's recipe sets CH_ENABLE_MODULE_IRRLICHT=OFF, VEHICLE=OFF,
# ROS=OFF; the projectchrono channel ships only core/fea/robot). This base image
# is a source-built PyChrono (python 3.10) that does expose pychrono.irrlicht,
# which the visualization layer requires. Pinned by digest for reproducibility.
FROM lucamarchiano/pychrono_simulator:1.0@sha256:a8843024d3162b272e5afcfe72c9d3e1c0289d1526c2bd67a73711c2f4a6e61a

USER root

# Xvfb (headless rendering), software GL and video tooling. The base already
# provides the Irrlicht/Chrono runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libglu1-mesa \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyChrono lives in /usr/local/share/chrono/python; add the package source
# without disturbing it. (The base pip/setuptools are too old for PEP 621
# installs, so we import from /work/src instead of pip-installing the package.)
ENV PYTHONPATH=/usr/local/share/chrono/python/:/work/src \
    LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /work
COPY . /work

# Drop the ROS entrypoint so commands run directly.
ENTRYPOINT []

CMD ["python", "-m", "drone6dof", "--vis", "none"]
