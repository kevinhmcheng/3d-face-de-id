#!/bin/sh

#Train/Test Classification Models for Biometric, Expression, Gender and Ethnicity Recognition
for i in 1 2 3 4
do
  for j in 1 2 3 4
  do
    python classification.py '2D' $i $j
    python classification.py 'Depth' $i $j
  done
done


#Train the De-identification Models
for k in 0 1 2 3 4 5 6
do
  python AE/de-identification.py '2D' $k
  python AE/de-identification.py 'Depth' $k
  python GAN/de-identification.py '2D' $k
  python GAN/de-identification.py 'Depth' $k
done


#Test the De-identification Performance with the Classification Models
for k in 2 3 4 5 6 12 13 14 15 16
do
  for i in 1 2 3 4
  do
    for j in 1 2 3 4
    do
      python classification_test_only.py '2D' $k $i $j
      python classification_test_only.py 'Depth' $k $i $j
    done
  done
done

