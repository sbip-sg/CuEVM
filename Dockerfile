

# add ethereum tests folder 
# compile binary
# compile dynamic library to be used by medusa-cuevm
# add medusa sample contracts
# add python scripts to run the ethereum tests in batch

FROM golang:latest AS golang-builder
ENV BUILD_DATE=20251002
RUN git clone https://github.com/cassc/goevmlab --depth 1
RUN cd goevmlab && \
  go build ./cmd/generic-fuzzer && \
  go build ./cmd/checkslow && \
  go build ./cmd/minimizer && \
  go build ./cmd/repro && \
  go build ./cmd/runtest && \
  go build ./cmd/tracediff && \
  go build ./cmd/traceview
RUN git clone https://github.com/ethereum/go-ethereum --depth 1
RUN cd go-ethereum && go run build/ci.go install -static ./cmd/evm

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic dependencies
RUN apt-get update

RUN apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    z3 \
    libz3-dev \
    libfontconfig1 \
    cmake

RUN apt-get install -y python3 python3-pip pipx
RUN apt-get install -y libcjson1  libcjson-dev

RUN pip3 install beautifulsoup4 --break-system-packages
COPY python-dependencies/slither-fix /tmp/slither-fix
COPY python-dependencies/crytic-compile-etherscan-offline /tmp/crytic-compile-etherscan-offline
COPY python-dependencies/solc-json-parser /tmp/solc-json-parser

RUN pip3 install --break-system-packages slither-analyzer==0.11.3
RUN pip3 install --break-system-packages crytic-compile==0.3.10
RUN pip3 install --break-system-packages matplotlib pandas numpy
RUN pip3 install solc-json-parser --break-system-packages
RUN pip3 install --force-reinstall --no-deps --break-system-packages /tmp/slither-fix
RUN pip3 install --force-reinstall --no-deps --break-system-packages /tmp/crytic-compile-etherscan-offline
RUN pip3 install --force-reinstall --no-deps --break-system-packages /tmp/solc-json-parser

RUN pip3 install lxml --break-system-packages
RUN pip3 install --break-system-packages py-solc-x solc-select && \
    mkdir -p /root/.solcx /root/.solc-select
RUN python3 - <<'PY'
import subprocess
from solcx import install_solc, get_installable_solc_versions

MIN_VERSION = (0, 4, 2)

def version_tuple(label: str) -> tuple[int, ...]:
    base = label.lstrip('v').split('-', 1)[0].split('+', 1)[0]
    return tuple(int(part) for part in base.split('.'))

versions: list[str] = []
seen = set()
for raw in get_installable_solc_versions():
    label = str(raw)
    numeric = version_tuple(label)
    if numeric >= MIN_VERSION and 'nightly' not in label:
        normalized = '.'.join(str(part) for part in numeric)
        if normalized not in seen:
            seen.add(normalized)
            versions.append(normalized)

versions.sort(key=version_tuple)

for version in versions:
    install_solc(version)

for version in versions:
    subprocess.check_call(["solc-select", "install", version])

if versions:
    subprocess.check_call(["solc-select", "use", versions[-1]])
PY

COPY . /opt/cuevm
RUN cmake -DBUILD_GO_LIBRARY=ON -DENABLE_EIP_3155=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY=86 -S /opt/cuevm -B /opt/cuevm/build \
    && cmake --build /opt/cuevm/build -j "$(nproc)" \
    && cp /opt/cuevm/build/libcuevm_go.so /usr/local/lib/
RUN rm -rf /opt/cuevm/build \
    && cmake -DBUILD_GO_LIBRARY=OFF -DENABLE_EIP_3155=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY="86;89" -S /opt/cuevm -B /opt/cuevm/build \
    && cmake --build /opt/cuevm/build -j "$(nproc)" \
    && cp /opt/cuevm/build/medusa-cuevm /usr/local/bin/

# todo update this once we release the medusa source code  
COPY medusa /usr/local/bin/

RUN mkdir /goevmlab
ENV PATH="${PATH}:/goevmlab"
COPY --from=golang-builder /go/goevmlab/generic-fuzzer /usr/local/bin
COPY --from=golang-builder /go/goevmlab/checkslow  /usr/local/bin
COPY --from=golang-builder /go/goevmlab/minimizer /usr/local/bin
COPY --from=golang-builder /go/goevmlab/repro /usr/local/bin
COPY --from=golang-builder /go/goevmlab/runtest /usr/local/bin
COPY --from=golang-builder /go/goevmlab/tracediff /usr/local/bin
COPY --from=golang-builder /go/goevmlab/traceview /usr/local/bin
COPY --from=golang-builder /go/go-ethereum/build/bin/evm /usr/local/bin

WORKDIR /tmp
RUN apt install -y wget unzip
RUN wget https://github.com/ethereum/tests/archive/refs/heads/shanghai.zip
RUN unzip shanghai.zip
RUN mv /tmp/tests-shanghai/GeneralStateTests /ethereum-tests-shanghai

RUN ldconfig

COPY scripts/run-ethtest-without-stateroot-comparison.py /usr/local/bin/run-ethtest-without-stateroot-comparison.py

# Set working directory
WORKDIR /app
