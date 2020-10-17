Class number : bao gom ca background.
VOC dataset : 
SSD300 -SSD512 : 
Tinh bounding box dua vao default box va Offset information(Thong qua transforms)
6 buoc co ban data di qua SSD :
1. resize buc anh ve 300x300
2. chuan bi default box (8732)
3. Truyen input anh vao mang SSD
4 . Lay ra bounding box cos confidence cao nhat 
5. Dung thuat toan non-maximun Suppression (NMS)
6. Chon mot cai threshold cho confidence 
    - High threshold : Tranh duoc vuoc detect nham
    - Low threshold : Tranh duoc detect thieu

