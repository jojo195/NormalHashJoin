for nradix in 9 10 11 12 13 14 15 16 17 18
do
    make NUM_RADIX_BITS=${nradix}
    wait
    ./bin/host_code -a PRH -n 4 -r 12800000 -s 12800000 > ./profile/PRH_nradix${nradix}.txt
    wait
    rm -rf ./bin
    wait
done