Mosaic of Classes is a visualization utility that provides
a possibility to create an image representation of subfolders content
of a given directory.

The resulting image is comprised of rows. Each row represents a
subdirectory and comprised of square tiles (its image content).

Mosaic of Classes facilities can be accessed either from another
Python script or from the command line.

If you want to use it from the command line,
(assuming calls from the create_mosaic.py directory)
a possible command could be something like this:

```
python create_mosaic.py input/dir/path output/path.jpg --labels
```

Such command will create an image that represents the content of
input/dir/path folder and puts folder names labels before each row.
For the complete list of commands run:

```
python create_mosaic.py -h
```
