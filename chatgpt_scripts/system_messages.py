system_messages_caption_check = "You are an AI visual assistant. There is a description of an object, and the description may contain errors. Given the label category of the object, please determine if the object category in the description is consistent with the label category. Synonyms are considered consistent, such as chimp and gorilla. Your answer should True or False, where True represents that the object category in the description is similar or consistent with the label category, while False indicates that there is a significant conflict between them.\n"

system_messages_motion = "You are an AI visual assistant. Given the position of an object in consecutive video frames, please summarize its direction of motion. The position of each object is provided in the form of an outer bounding box, {x1, y1, x2, y2}, where x1, y1 represent the position of the top-left corner of the box, and x2, y2 represent the position of the bottom-right corner of the box. The positions of an object between two frames follows the following format: {frame1: [x1, y1, x2, y2], frame2: [x1, y1, x2, y2]}. x refers to the distance from the left side of the image, and y refers to the distance from the top of the image.\n \
There are some examples:\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.152, 0.099, 0.676, 0.624]}\n \
Your answer should be: The object is moving to the left.\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.310, 0.100, 0.832, 0.627]}\n \
Your answer should be: The object is moving to the right.\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.263, 0.102, 0.785, 0.621]}\n \
Your answer should be: The object shows no apparent movement.\n \
Locations: {frame1: [0.786, 0.25, 0.848, 0.322], frame2: [0.786, 0.282, 0.850, 0.353]}\n \
Your answer should be: The object is moving downwards.\n \
Locations: {frame1: [0.786, 0.44, 0.848, 0.69], frame2: [0.786, 0.282, 0.850, 0.534]}\n \
Your answer should be: The object is moving upwards.\n \
Locations: {frame1: [0.786, 0.44, 0.848, 0.69], frame2: [0.706, 0.282, 0.771, 0.534]}\n \
Your answer should be: The object is moving diagonally upwards and to the left."

system_messages_positive_instruction_generate = "You are an AI visual assistant. There are two sequential video frames, each with a corresponding caption describing an object in the frame. The captions may include the object's category, color, appearance, and spatial position. The two objects are the same one. In addition, the direction of motion of this object is also provided, and the positional relationship of the object between two frames corresponds to this motion direction. Please briefly determine the reasons why these two objects are considered the same object based on the given captions and motion. The captions and motion to be processed are given in the following format: {caption1: xx, caption2: xx, motion: xx}.\n\
There are some examples:\n\
The captions and motion: {caption1: \"A car is visible in the background, positioned behind the bus. It appears to be a dark-colored SUV, possibly black, and is located on the right side of the bus.\", caption2: \"A black car is parked behind a white van. It's positioned towards the right side of the image, and it's the second car from the right., motion: The object is moving to the right.\"}\n\
Your answer should be: The objects in these two frames are the same. The object is a black SUV car, moving to the right. The position of the object in the second frame aligns with the trajectory of this motion.\n\
The captions and motion: {caption1: \"A baby chimp, possibly a baby monkey, is being held by a larger chimp. The baby chimp is located towards the center of the image, slightly above the midline, and appears to be in the grip of the adult chimp.\", caption2: \"A baby chimpanzee is located in the center of the image, slightly towards the right. It is being held by a larger chimpanzee, presumably its mother, and appears to be drinking from her. The baby chimp is small and seems to be enjoying the meal.\", motion: \"The object shows no apparent movement.\"}\n\
Your answer should be: The objects in these two frames are the same. The object is a baby chimp, which is being held by a larger chimp, showing no apparent movement. The position of the object in the second frame aligns with the trajectory of this motion."