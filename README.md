
# Face swap using dlib and OpenCV

### Installation guide

Clone repository:
```
git clone https://github.com/nWhovian/AnastasiaTest.git
cd path/to/repository
```
The ```face_swap.ipynb``` file contains a detailed solution with illustrations, python scripts contain more structured, readable code with an improved triangulation approach

[comment]: <> (Download the docker image:)

[comment]: <> (```)

[comment]: <> (docker-compose pull)

[comment]: <> (```)

[comment]: <> (Start and run the app:)

[comment]: <> (```)

[comment]: <> (docker-compose up)

[comment]: <> (```)

[comment]: <> (Access the running container:)

[comment]: <> (```)

[comment]: <> ( docker exec -it car_tracker_app_1 bash)

[comment]: <> (```)
Run the algorithm:
```
python main.py --img1 assets/img_3.jpeg --img2 assets/img_4.jpeg
```
Optional arguments: ```--out results --predictor_path assets/shape_predictor_68_face_landmarks.dat```

The output files are located here by default: path/to/repository/results