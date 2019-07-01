#!/bin/bash

counter=0
for i in {0..4}
do
    if [ $i -eq 0 ]
    then
        mapping="oversampling"
    elif [ $i -eq 1 ]
    then
        mapping="rebinning"
    elif [ $i -eq 2 ]
    then
        mapping="nearest_interpolation"
    elif [ $i -eq 3 ]
    then
    mapping="bilinear_interpolation"
    elif [ $i -eq 4 ]
    then
    mapping="bicubic_interpolation"
    fi
    for j in {0..2}
    do
        if [ $j -eq 0 ]
        then
            tel_type="LSTCam"
        elif [ $j -eq 1 ]
        then
        tel_type="FlashCam"
        elif [ $j -eq 2 ]
        then
        tel_type="DigiCam"
        fi
        if [ $counter -lt 10 ]
        then
            echo run0$counter $tel_type$mapping
            mv run0$counter $tel_type$mapping
        else
            echo run$counter $tel_type$mapping
            mv run$counter $tel_type$mapping
        fi
        counter=$((counter+1))
    done
done
