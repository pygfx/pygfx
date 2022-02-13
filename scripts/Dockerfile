# Use ubuntu as the base image (it's the only distro
# lavapipe is available for prebuilt)
FROM docker.io/library/ubuntu:20.04

# Install python3.9 and lavapipe
RUN apt-get update -y -qq --fix-missing \
    && apt-get install -y -qq --no-install-recommends curl software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:oibaf/graphics-drivers -y \
    && apt-get update -y -qq \
    && apt-get install -y -qq --no-install-recommends python3.9 python3.9-distutils libegl1-mesa libgl1-mesa-dri libxcb-xfixes0-dev mesa-vulkan-drivers \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.9 get-pip.py \
    && apt-get remove -y -qq curl software-properties-common \
    && apt-get autoremove -y -qq \
    && rm -rf get-pip.py /var/lib/apt/lists/*

# Set the working directory, and install development dependencies
WORKDIR /app
COPY dev-requirements.txt .
RUN pip install -r dev-requirements.txt \
    && pip cache purge --no-input

# Install pygfx in editable mode
# (we only copy the files that are actually needed for the install to succeed)
COPY pygfx/__init__.py pygfx/__init__.py
COPY setup.py README.md LICENSE .
RUN pip install -e . \
    # workaround for pip editable install issue
    && pip install --no-deps --install-option="--install-dir=/usr/local/lib/python3.9/dist-packages" -e . \
    && pip cache purge --no-input

# Configure pygfx
ENV WGPU_FORCE_OFFSCREEN=true

# Add all the other paths that don't influence installation
COPY pygfx/ pygfx/
COPY tests/ tests/
COPY examples/ examples/
COPY docs/ docs/
COPY conftest.py setup.cfg .
