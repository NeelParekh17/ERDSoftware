# Example data from your labels
labels = [
    {"type": "Entity", "x_center": 1376.7169189453125, "y_center": 1344.7802734375, "width": 287.930419921875, "height": 113.0391845703125},
    {"type": "Entity", "x_center": 261.23773193359375, "y_center": 993.618896484375, "width": 298.603759765625, "height": 108.58233642578125},
    {"type": "Relationship", "x_center": 1032.9244384765625, "y_center": 421.0442199707031, "width": 271.98651123046875, "height": 195.63226318359375},
    {"type": "Entity", "x_center": 1015.6732788085938, "y_center": 983.1307983398438, "width": 210.8978271484375, "height": 111.9283447265625},
    {"type": "Relationship", "x_center": 1826.541015625, "y_center": 1221.2232666015625, "width": 297.2864990234375, "height": 156.719482421875},
    {"type": "Relationship", "x_center": 1035.72705078125, "y_center": 739.4268798828125, "width": 288.1729736328125, "height": 168.51824951171875},
    {"type": "Attribute", "x_center": 1573.028076171875, "y_center": 1543.98583984375, "width": 143.8900146484375, "height": 85.9334716796875},
    {"type": "Relationship", "x_center": 1128.5435791015625, "y_center": 1833.093505859375, "width": 261.64129638671875, "height": 151.7293701171875}
]

# Formatting function
def format_label(label):
    return f'{{"type": "{label["type"]}", "x_center": {label["x_center"]}, "y_center": {label["y_center"]}, "width": {label["width"]}, "height": {label["height"]}}}'

# Output all labels in desired format
formatted_labels = [format_label(label) for label in labels]

# Join the formatted strings with newlines
formatted_output = ",\n".join(formatted_labels)

print(formatted_output)
