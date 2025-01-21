put video with name "test_video.MP4" in data/ folder 

sudo docker build -t my-test-task .

sudo docker run -it -v "$(pwd)"/data:/workspace --rm --gpus all my-test-task
