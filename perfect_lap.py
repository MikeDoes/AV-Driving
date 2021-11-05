# in command prompt, type "pip install pynput" to install pynput.
from pynput.keyboard import Key, Controller
from time import sleep



sleep(2)
keyboard = Controller()
up = Key.up
down = Key.down
right = Key.right
left = Key.left

#Straight Line
keyboard.press(up)
sleep(2.9)

#Circular Turn
keyboard.press(right)
sleep(3)
#slow down before inflection and sharp turn
keyboard.release(up)
sleep(1)
#Inflection Turn
keyboard.release(right)
keyboard.press(left)
sleep(1.98)

#Sharp Turn
keyboard.release(left)
keyboard.press(right)
sleep(1.6)

#Accelerate back through the sharp turn
keyboard.press(up)
sleep(0.5)
keyboard.release(up)
sleep(1)


#Straight line
keyboard.release(right)



keyboard.release(up)


