# Silent-Speech-Sleuth (S3)

A few weeks ago, La Liga hired a professional lip reader to decide whether Jude Bellingham called Mason Greenwood a “rapist” after the two players clashed during a football (not soccer) match.
Now I am not going to look much into the details of the incident, instead what I am going to try is to build a deep learning model that is able to read lips since not many of us are lunatics (only lunatics can read lips), or has pockets deep enouhg for a pro lip reader (ours, yes, mine and yours are only as deep because they are empty).

Lip reading, a skill crucial for those with hearing impairments, has long been a challenging task due to its complexity and nuances. However, with the advent of advanced technologies like LipNet, a deep learning model designed to interpret lip movements, there is newfound hope for enhanced communication accessibility. LipNet showcases the potential of artificial intelligence in bridging communication gaps and empowering individuals with hearing impairments to engage more fully in conversations. As we delve deeper into the realm of assistive technologies, LipNet stands as a promising tool, offering a glimpse into a future where communication barriers are minimized, and inclusivity thrives.

I'll be using a range of technologies; OpenCV to read our videos and TensorFlow to build the model.

# Workflow

1. Build Data Loading Function
2. Create Data Pipeline
3. Design the Deep Neural Network
4. Setup Training Options and Train
5. Make a Prediction
6. Test on a Sample Video
7. Create an Interface for Realtime Interaction With The Model
