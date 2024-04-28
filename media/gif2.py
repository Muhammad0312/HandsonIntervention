import subprocess

# Path to your MP4 video
video_path = "intervention_perception.mp4"

# Output GIF filename
output_gif = "intervention.gif"

# Command to convert MP4 to GIF with ffmpeg (reduced size)
command = ["ffmpeg", "-i", video_path, "-vf", "scale=600:-1,fps=10", output_gif]

# Run the ffmpeg command
subprocess.run(command)
