#!/usr/bin/gnuplot --persist

#If need to test for arguments passing
#print "input file        : ", ARG1
#print "ouput file     : ", ARG2

#Set the output file 
set term png size 1000, 800 enhanced font "Arial,18"
set output ARG2
set border 31 lw 3.0
set tics out 
set xtics nomirror


#Format and plotting
set key bottom right
set xlabel "{/:Bold Step}"
#set xrange [0.0:1000]
set yrange [-0.5:2.5]
set ytics ("Fluid" 0.0, "Deffective" 1.0, "Crystal" 2.0)
set ytics nomirror


plot ARG1 u ($1):($3) title "SNN predicted" pointtype 6 pointsize 2.0 lc 8