![Simple 2D animation software.](https://i.imgur.com/npR8luu.png)

## Quickstart
```
virtualenv -p python2.7 ./animation-env
source ./animation-env/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
pip install -r requirements.txt
python main.py
```

## Features
### Currently supported
- customizable background image
- customizable foreground images, trace in real time the trajectory you want them to follow
- optional spray trail
- ricochet loop option
- export animation to GIF (`imageio`) or MOV (`OpenCV`)
- scrub through animation using timeline slider

### Under development
- handle non-RGB/RGBA images
- "no loop" option for each foreground object
- camera transforms, including orbit
- full-object position/rotation/scaling interpolation between different keypoints (tweening)
- lasso select object components to animate individually
- 3D workflow: modify transforms in each coordinate plane, mix 2D and 3D objects
- stylization for rendering (e.g. non-photorealistic)
- automatic background removal
- standard behavior for "background image dims != provided dims" case
- automatically smooth paths that the user draws
- click on object to select for use in tracing
- hide/show different layers at different timesteps
- import new images (foreground, background) in-app

## Controls
- `s`: switch active foreground object
- `space`: clear trace for active foreground object
- `tab`: hide control bar
- `hold cursor and move`: record trace for active foreground object, if one doesn't already exist
