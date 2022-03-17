# Convert qr-code detection dataset to yolo

`python to_yolo.py --input {label_dir} --image {image_dir} --ann {annotation_dir}`
### Input 
1. [absolute horizontal position] - in pixels
2. [absolute vertical position] - in pixels
3. [radius of the circumcircle] - in pixels
4. [FLAG - rotated] - 0,1
5. [FLAG - blurred] - 0,1
6. [FLAG - perspective transform] - 0,1
7. [FLAG - extreme lighting] - 0,1
8. [annotated qrcode bitmap name] - string (without the extension .png - has to be added when parsing)

[Data Source](http://www.fit.vutbr.cz/research/groups/graph/pclines/pub_page.php?id=2012-SCCG-QRtiles)

### Output
Yolo format (class x_center y_center width height)

## Check if it works
`python to_yolo.py --label {label_dir} --image {image_dir} --save {save_dir}`

![plot](./demo/demo.png)