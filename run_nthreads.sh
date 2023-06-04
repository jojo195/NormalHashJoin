for nthread in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
    make NUM_RADIX_BITS=14
    wait
    ./bin/host_code -a PRH -n ${nthread} -r 12800000 -s 12800000 > ./profile/PRH_nthread${nthread}.txt
    wait
    rm -rf ./bin
    wait
done