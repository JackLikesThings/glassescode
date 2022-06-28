# SPDX-FileCopyrightText: 2021 Phil Burgess for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
AUDIO SPECTRUM LIGHT SHOW for Adafruit EyeLights (LED Glasses + Driver).
Uses onboard microphone and a lot of math to react to music.
"""

from array import array
from math import log
from time import monotonic
from supervisor import reload
import board
from audiobusio import PDMIn
from busio import I2C
import adafruit_is31fl3741
from adafruit_is31fl3741.adafruit_ledglasses import LED_Glasses
from rainbowio import colorwheel
from ulab import numpy as np
from ulab.scipy.signal import spectrogram


# import random
import time
# import board
# import busio
import digitalio
import displayio
import framebufferio
import is31fl3741
from adafruit_is31fl3741.is31fl3741_PixelBuf import IS31FL3741_PixelBuf
from adafruit_is31fl3741.led_glasses_map import (
    glassesmatrix_ledmap_no_ring,
    left_ring_map_no_inner,
    right_ring_map_no_inner,
)
from adafruit_display_text import label
from adafruit_bitmap_font import bitmap_font
from adafruit_led_animation.animation.chase import Chase
from adafruit_debouncer import Debouncer

# List of possible messages to display. Randomly chosen
MESSAGES = (
    "GO TEAM GO",
    "WE ARE NUMBER 1",
    "I LIKE THE HALFTIME SHOW",
)

# Colors used for the text and ring lights
BLUE_TEXT = (0, 20, 255)
BLUE_RING = (0, 10, 120)
YELLOW_TEXT = (220, 210, 0)
YELLOW_RING = (150, 140, 0)

def Score(text, color, ring_color, repeat):
    """Show a scrolling text message and animated effects on the eye rings.
    The messages scrolls left to right, then right to left while the eye rings
    are animated using the adafruit_led_animation library."""

    # Set up a led animation chase sequence for both eyelights
    chase_left = Chase(left_eye, speed=0.11, color=ring_color, size=8, spacing=4)
    chase_right = Chase(right_eye, speed=0.07, color=ring_color, size=8, spacing=4)

    text_area.text = text
    text_area.color = color

    x = display.width
    text_area.x = x

    width = text_area.bounding_box[2]

    for _ in range(repeat):
        # Scroll the text left and animate the eyes
        while x != -width:
            x = x - 1
            text_area.x = x
            chase_left.animate()
            chase_right.animate()
            time.sleep(0.008)  # adjust to change scrolling speed
        # Scroll the text right and animate the eyes
        while x != display.width:
            x = x + 1
            text_area.x = x
            chase_left.animate()
            chase_right.animate()
            time.sleep(0.008)  # adjust to change scrolling speed


# Set up the top button used to trigger a special message when pressed
switch_pin = digitalio.DigitalInOut(board.SWITCH)
switch_pin.direction = digitalio.Direction.INPUT
switch_pin.pull = digitalio.Pull.UP
switch = Debouncer(switch_pin)


# FFT/SPECTRUM CONFIG ----

fft_size = 256  # Sample size for Fourier transform, MUST be power of two
spectrum_size = fft_size // 2  # Output spectrum is 1/2 of FFT result
# Bottom of spectrum tends to be noisy, while top often exceeds musical
# range and is just harmonics, so clip both ends off:
low_bin = 10  # Lowest bin of spectrum that contributes to graph
high_bin = 75  # Highest bin "


# HARDWARE SETUP ---------

# Manually declare I2C (not board.I2C() directly) to access 1 MHz speed...
i2c = I2C(board.SCL, board.SDA, frequency=1000000)

# Initialize the IS31 LED driver, buffered for smoother animation
glasses = LED_Glasses(i2c, allocate=adafruit_is31fl3741.MUST_BUFFER)
glasses.show()  # Clear any residue on startup
glasses.global_current = 5  # Not too bright please

# Initialize mic and allocate recording buffer (default rate is 16 MHz)
mic = PDMIn(board.MICROPHONE_CLOCK, board.MICROPHONE_DATA, bit_depth=16)
rec_buf = array("H", [0] * fft_size)  # 16-bit audio samples


# In this example scale is set to True. When True the logical display is
# three times the physical display size and scaled down to allow text to
# look more natural for small display sizes. Hence the display is created
# as 54x15 when the physical display is 18x5.
#
# i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)
is31 = is31fl3741.IS31FL3741(i2c=i2c)
is31_framebuffer = is31fl3741.IS31FL3741_FrameBuffer(
    is31, 54, 15, glassesmatrix_ledmap_no_ring, scale=True, gamma=True
)
display = framebufferio.FramebufferDisplay(is31_framebuffer, auto_refresh=True)

# Set up the left and right eyelight rings
# init is set to False as the IS31FL3741_FrameBuffer has already initialized the IS31FL3741 driver
left_eye = IS31FL3741_PixelBuf(
    is31, left_ring_map_no_inner, init=False, auto_write=False
)
right_eye = IS31FL3741_PixelBuf(
    is31, right_ring_map_no_inner, init=False, auto_write=False
)

# Dim the display. Full brightness is BRIGHT
is31_framebuffer.brightness = 0.2

# Load the font to be used - scrolly only has upper case letters
font = bitmap_font.load_font("/fonts/scrolly.bdf")

# Set up the display elements
text_area = label.Label(font, text="", color=(0, 0, 0))
text_area.y = 8
group = displayio.Group()
group.append(text_area)
display.show(group)



# FFT/SPECTRUM SETUP -----

# To keep the display lively, tables are precomputed where each column of
# the matrix (of which there are few) is the sum value and weighting of
# several bins from the FFT spectrum output (of which there are many).
# The tables also help visually linearize the output so octaves are evenly
# spaced, as on a piano keyboard, whereas the source spectrum data is
# spaced by frequency in Hz.
column_table = []

spectrum_bits = log(spectrum_size, 2)  # e.g. 7 for 128-bin spectrum
# Scale low_bin and high_bin to 0.0 to 1.0 equivalent range in spectrum
low_frac = log(low_bin, 2) / spectrum_bits
frac_range = log(high_bin, 2) / spectrum_bits - low_frac

for column in range(glasses.width):
    # Determine the lower and upper frequency range for this column, as
    # fractions within the scaled 0.0 to 1.0 spectrum range. 0.95 below
    # creates slight frequency overlap between columns, looks nicer.
    lower = low_frac + frac_range * (column / glasses.width * 0.95)
    upper = low_frac + frac_range * ((column + 1) / glasses.width)
    mid = (lower + upper) * 0.5  # Center of lower-to-upper range
    half_width = (upper - lower) * 0.5  # 1/2 of lower-to-upper range
    # Map fractions back to spectrum bin indices that contribute to column
    first_bin = int(2 ** (spectrum_bits * lower) + 1e-4)
    last_bin = int(2 ** (spectrum_bits * upper) + 1e-4)
    bin_weights = []  # Each spectrum bin's weighting will be added here
    for bin_index in range(first_bin, last_bin + 1):
        # Find distance from column's overall center to individual bin's
        # center, expressed as 0.0 (bin at center) to 1.0 (bin at limit of
        # lower-to-upper range).
        bin_center = log(bin_index + 0.5, 2) / spectrum_bits
        dist = abs(bin_center - mid) / half_width
        if dist < 1.0:  # Filter out a few math stragglers at either end
            # Bin weights have a cubic falloff curve within range:
            dist = 1.0 - dist  # Invert dist so 1.0 is at center
            bin_weights.append(((3.0 - (dist * 2.0)) * dist) * dist)
    # Scale bin weights so total is 1.0 for each column, but then mute
    # lower columns slightly and boost higher columns. It graphs better.
    total = sum(bin_weights)
    bin_weights = [
        (weight / total) * (0.8 + idx / glasses.width * 1.4)
        for idx, weight in enumerate(bin_weights)
    ]
    # List w/five elements is stored for each column:
    # 0: Index of the first spectrum bin that impacts this column.
    # 1: A list of bin weights, starting from index above, length varies.
    # 2: Color for drawing this column on the LED matrix. The 225 is on
    #    purpose, providing hues from red to purple, leaving out magenta.
    # 3: Current height of the 'falling dot', updated each frame
    # 4: Current velocity of the 'falling dot', updated each frame
    column_table.append(
        [
            first_bin - low_bin,
            bin_weights,
            colorwheel(225 * column / glasses.width),
            glasses.height,
            0.0,
        ]
    )
# print(column_table)


# MAIN LOOP -------------

dynamic_level = 10  # For responding to changing volume levels
frames, start_time = 0, monotonic()  # For frames-per-second calc

while True:
    # The try/except here is because VERY INFREQUENTLY the I2C bus will
    # encounter an error when accessing the LED driver, whether from bumping
    # around the wires or sometimes an I2C device just gets wedged. To more
    # robustly handle the latter, the code will restart if that happens.
    
    switch.update()

    # If the switch has been pressed interrupt start a special message
    if not switch.value:
        Score("JONK", YELLOW_TEXT, BLUE_RING, 1)
   
    
    try:
        mic.record(rec_buf, fft_size)  # Record batch of 16-bit samples
        samples = np.array(rec_buf)  # Convert to ndarray
        # Compute spectrogram and trim results. Only the left half is
        # normally needed (right half is mirrored), but we trim further as
        # only the low_bin to high_bin elements are interesting to graph.
        spectrum = spectrogram(samples)[low_bin : high_bin + 1]
        # Linearize spectrum output. spectrogram() is always nonnegative,
        # but add a tiny value to change any zeros to nonzero numbers
        # (avoids rare 'inf' error)
        spectrum = np.log(spectrum + 1e-7)
        # Determine minimum & maximum across all spectrum bins, with limits
        lower = max(np.min(spectrum), 4)
        upper = min(max(np.max(spectrum), lower + 6), 20)

        # Adjust dynamic level to current spectrum output, keeps the graph
        # 'lively' as ambient volume changes. Sparkle but don't saturate.
        if upper > dynamic_level:
            # Got louder. Move level up quickly but allow initial "bump."
            dynamic_level = upper * 0.7 + dynamic_level * 0.3
        else:
            # Got quieter. Ease level down, else too many bumps.
            dynamic_level = dynamic_level * 0.5 + lower * 0.5

        # Apply vertical scale to spectrum data. Results may exceed
        # matrix height...that's OK, adds impact!
        data = (spectrum - lower) * (7 / (dynamic_level - lower))

        for column, element in enumerate(column_table):
            # Start BELOW matrix and accumulate bin weights UP, saves math
            first_bin = element[0]
            column_top = glasses.height + 1
            for bin_offset, weight in enumerate(element[1]):
                column_top -= data[first_bin + bin_offset] * weight

            if column_top < element[3]:  #       Above current falling dot?
                element[3] = column_top - 0.5  # Move dot up
                element[4] = 0  #                and clear out velocity
            else:
                element[3] += element[4]  #      Move dot down
                element[4] += 0.2  #             and accelerate

            column_top = int(column_top)  #      Quantize to pixel space
            for row in range(column_top):  #     Erase area above column
                glasses.pixel(column, row, 0)
            for row in range(column_top, 5):  #  Draw column
                glasses.pixel(column, row, element[2])
            glasses.pixel(column, int(element[3]), 0xE08080)  # Draw peak dot

        glasses.show()  # Buffered mode MUST use show() to refresh matrix

        frames += 1
        # print(frames / (monotonic() - start_time), "FPS")


    except OSError:  # See "try" notes above regarding rare I2C errors.
        print("Restarting")
        reload()
