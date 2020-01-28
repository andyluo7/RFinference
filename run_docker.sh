tagname="Xl_rf_inference"
imagename="alveo-xrt-base"


user=`whoami`

HERE=`dirname $(readlink -f $0)`

mkdir -p $HERE/share
chmod -R a+rwx $HERE/share

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
echo "Found xclmgmt driver(s) at ${xclmgmt_driver}"
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
echo "Found render driver(s) at ${render_driver}"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

docker run \
    -it \
    --rm \
    --env H2O_XELERA=H2O \
    -p 12345:12345 \
    -v $HERE/scoring-pipeline/:/app/scoring-pipeline \
    -v $HERE/license/:/opt/h2oai/dai/home/.driverlessai/ \
    -v $HERE/run_standalone_benchmark.py:/app/run_standalone_benchmark.py \
    -v $HERE/run_standalone_benchmark.sh:/app/run_standalone_benchmark.sh \
    -v $HERE/run_Xl_benchmark_single.py:/app/run_Xl_benchmark_single.py \
    $docker_devices \
    --name xelera-${imagename} \
    ${imagename}:${tagname} \
