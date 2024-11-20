# image-registration
Julia source code to stabilize and clean up videos

How to use this code, its initial setup is convoluded as a result of many hands involved in its creation and operation.

Require languages: python and Julia

Begin by placing in the data folder each video into its own folder along with a copy of the registration.ipynb. 

```
- project
  - data
    - video1
        - video1.mp4
        - registration.ipynb
    - video2
        - video2.mp4
        - registration.ipynb
    - video3
        - video3.mp4
        - registration.ipynb
```

Initiate the Julia environment
```console
foo@bar:~ $ cd path/to/file
foo@bar:~ $ julia
julia> ]
pkg> activate .
(project name) pkg> instantiate
```
Leave this terminal window opened DO NOT TOUCH IT AFTER THIS 

Open a new terminal window and launch jupyter notebook
```console
foo@bar:~ $ cd path/to/file
foo@bar:~ $ jupyter noteboo
```
In the jupyter notebook run the cells necessary small changes will be required depending on the video.
  
