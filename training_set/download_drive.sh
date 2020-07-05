#!/bin/bash

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$1 -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $2 'https://docs.google.com/uc?export=download&id='$1'&confirm='$(<confirm.txt)


rm -f confirm.txt cookies.txt