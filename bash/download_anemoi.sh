target_dir=$1

if [ ! -d anemoi-core ]; then
    echo "Cloning anemoi-core"
    git clone https://github.com/ecmwf/anemoi-core.git \
        --branch ds-collab \
        "$target_dir"/anemoi-core
fi

if [ ! -d anemoi-inference ]; then
    echo "Cloning anemoi-inference"
    git clone https://github.com/ecmwf/anemoi-inference.git \
        --branch feat/core-c3s2-361b \
        "$target_dir"/anemoi-inference
fi
