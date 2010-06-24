#!/bin/bash

echo "*********************************************"
echo " Publishing documentation on zion "
echo "*********************************************"



scp -r _build/html/* rreyes@zion.pcg.ull.es:/home/webs/portales/sites/llc.pcg.ull.es/files/llc-content/devel/doc



echo " Documentation has been updated "
