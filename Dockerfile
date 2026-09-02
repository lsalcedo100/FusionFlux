# One-click reproduction of every number under results/.
#
# `make reproduce` regenerates the whole study from raw data and fails if any
# committed artifact moved. Running it needs a specific environment: Python 3.10
# or newer (stock macOS still ships 3.9, which fails the type check and four
# tests), the pinned dependency set every reported number was generated in, and
# two third-party datasets fetched on demand. Getting that wrong is the most
# likely reason a reader's numbers would differ from the committed ones, and it
# would look like irreproducibility rather than like a mismatched environment.
#
# This image removes that variable:
#
#     docker build -t fusionflux .
#     docker run --rm fusionflux                 # regenerate and diff, ~20 minutes
#     docker run --rm fusionflux make check      # lint, types, the test suite
#     docker run --rm -it fusionflux bash        # poke at it
#
# The datasets are fetched at run time rather than baked in. They are
# third-party and not redistributable here (the study pins them by SHA-256 and
# verifies on every load), so an image carrying them would be republishing
# someone else's data.

FROM python:3.12-slim-bookworm

# git is present so `make reproduce`'s `git rev-parse HEAD` runs cleanly rather
# than printing "command not found" before its fallback. The build context
# excludes .git (see .dockerignore), so that call reports no-git and the target
# takes its cautious branch, which is correct here: there is no working tree to
# restore a snapshot over.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# The whole tree before installing, rather than the usual dependency-first
# layer. `pip install -e .` runs setup.py's build hook, which copies
# results/predictor.json into the package so `fusionflux predict` has its
# coefficients, and setuptools needs the `fusionflux` package directory to exist
# to install it at all. Splitting the copy to cache the dependency layer would
# mean installing before either is present, which fails.
COPY . .

# The constraints file is the environment every number under results/ was
# generated in, so it is pinned to rather than resolved fresh. That is the whole
# point of building this image.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[dev]" -c constraints.txt

# Non-interactive matplotlib, and a writable home for any cache the libraries
# want, so the image also runs read-only or as a non-root user.
ENV MPLBACKEND=Agg \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

# The default is the claim the repository makes about itself: that results/
# still follows from the raw data. Override with any other make target.
CMD ["make", "reproduce"]
