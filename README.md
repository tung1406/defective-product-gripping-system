# Overall

Drop multiple garment buttons on the conveyor then detect the color-defective one and extract its coordinates. Finally, use Arduino to grip it on the conveyor belt.

# System model

![mattren - Copy](https://github.com/tung1406/defective-product-gripping-system/assets/105976089/2324c1f3-5b40-4c5a-b49e-23bd57d0c7e1)

1. Scanning booth and monitor screen.
2. Controller system and button tray.
3. Gripping system

Buttons are dropped at the scanning booth (1), then the computer detects what kind of button (standard, size defective, color defective). If the button is color defective, the system (2) will control the gripping system (3) to grip that button and drop it on the tray.

# User Interface
![giaodien - Copy](https://github.com/tung1406/defective-product-gripping-system/assets/105976089/51cccd20-2d2b-41d9-9ee6-f9036d9e6a11)

The left frame is camera screen. The right one is tracking screen, which is used to track the buttons and determine if they are color defective. At the bottom, you can find the name of the project, the counter to track the quantity of each button type, and the RESET and EXIT buttons.

# Detecting button

![giaodien2](https://github.com/tung1406/defective-product-gripping-system/assets/105976089/fb387905-859c-4d42-8cc8-517f6aa9af89)

There are three lines: purple, cyan, and green. The purple line is used for tracking the buttons. The cyan one is used for determining color-defective buttons. The last line is used for counting the quantity.

![giaodien3](https://github.com/tung1406/defective-product-gripping-system/assets/105976089/eb7dca14-3b07-4d03-a75c-0872b14e74f2)

If the color-defective buttons cross the cyan line, those buttons will be shown as "TARGET".
