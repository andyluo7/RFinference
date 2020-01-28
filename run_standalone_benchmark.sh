
touch results.txt
touch intermediate_results.txt
touch raw_results.txt

python3 run_standalone_benchmark.py Xl_rf_inference.xclbin ./ > raw_results.txt

more raw_results.txt | grep "^LOGINFO" > intermediate_results.txt
sed -e 's/^\w*\ *//' intermediate_results.txt > results.txt

rm -f intermediate_results.txt
rm -f raw_results.txt
