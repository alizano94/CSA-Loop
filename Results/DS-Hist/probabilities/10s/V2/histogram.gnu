#!/usr/bin/gnuplot --persist

#If need to test for arguments passing
#print "input file        : ", ARG1
#print "ouput file     : ", ARG2

#Set the output file 
set term png size 1000, 800 enhanced font "Arial,18"
set style line 1 lc 'red' lt 1 lw 2
set style data histograms
set style fill solid
set output ARG2
set border 31 lw 3.0
set tics out 
set xtics nomirror




#Format and plotting
unset key
set ytics nomirror
set yrange [0:]

plot ARG1 using 0:($2/595):(0.7):xtic(1) w boxes ls 1