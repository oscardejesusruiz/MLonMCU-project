import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

# 1. Load the CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (_, _) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Create the display window
window_name = "CIFAR-10 Manual Viewer"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 400, 400)

print("\nInstructions:")
print("-> Press [SPACE] to show the next image.")
print("-> Press [ESC] at any time to quit.")

# 3. Loop successively through the images
i = 0
while i < len(X_train):
    image = X_train[i]

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize and add label text
    label = class_names[y_train[i][0]]
    display_img = cv2.resize(image_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display_img, f"Img #{i}: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image on screen
    cv2.imshow(window_name, display_img)

    # 4. Wait indefinitely until the user presses a key.
    #    cv2.waitKey(0) blocks until a key is pressed, returning its code.
    #    Keep polling so the window stays responsive (close button works).
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 32:           # SPACE → advance to next image
            i += 1
            break
        if key == 27:           # ESC → quit
            print("Exiting...")
            i = len(X_train)    # break the outer loop
            break
        # Window closed via the OS close button → quit
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            i = len(X_train)
            break

# Clean up windows when done
cv2.destroyAllWindows()
