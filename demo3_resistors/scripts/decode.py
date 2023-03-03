def decode(bands):

    # Init
    color = ['k', 'z', 'r', 'o', 'y','g', 'b', 'v', 'x', 'w']
    value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ohm, multiplier, unit = '', '', ' ohms'

    # Decode third band
    for i in range(len(color)):
        if bands[2] == color[i]:
            multiplier = 10**value[i]

    # Decode secodn and first band
    for i in range(len(color)):
        if bands[1] == color[i]:
            bands_one = value[i]
        if bands[0] == color[i]:
            bands_zero = value[i]

    # Calculate result
    res = int(str(bands_zero) + str(bands_one))*multiplier

    # Convert
    if res >= 1000000:
        res = str(res/1000000)
        if res[-1] == '0':
            res = res[0:-2] + 'M'
        else: res = res + 'M'
    elif res >= 1000:
        res = str(res/1000)
        if res[-1] == '0':
            res = res[0:-2] + 'k'
        else: res = res + 'k'

    # Finish
    ohm = str(res) + unit 

    return ohm