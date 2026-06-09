import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

# 1. Load the CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (_, _) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Create the display window
window_name = "CIFAR-10 Auto Viewer"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 400, 400)

print("\nInstructions:")
print("-> Images will automatically change every 2 seconds.")
print("-> Press [ESC] at any time to quit.")

# 3. Loop successively through the images
for i, image in enumerate(X_train):
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize and add label text
    label = class_names[y_train[i][0]]
    display_img = cv2.resize(image_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display_img, f"Img #{i}: {label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image on screen
    cv2.imshow(window_name, display_img)
    
    # 4. Wait for key press
    # If the user presses a key during this window, it captures it.
    key = cv2.waitKey(0)
    
    # If user presses ESC (ASCII code 27), break the loop and quit
    if key == 27:
        print("Exiting...")
        break
    # If user presses SPACE (ASCII code 32), continue to next image
    elif key == 32:
        print(f"Showing next image (image #{i})")
        continue

# Clean up windows when done
cv2.destroyAllWindows()