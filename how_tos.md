## How to remove commit history?
1. Checkout

   ```bash
   $ git checkout --orphan latest_branch
   ```

2. Add all the files

   ```bash
   $ git add -A
   ```

3. Commit the changes

   ```bash
   $ git commit -am "commit message"
   ```

4. Delete the branch

   ```bash
   $ git branch -D master
   ```

5. Rename the current branch to master

   ```bash
   $ git branch -m master
   ```

6. Finally, force update your repository

   ```bash
   $ git push -f origin master
   ```

## How to upload packages to PyPi
1.
```bash
$ python setup.py sdist
```
2.
```bash
$ python setup.py bdist_wheel
```
3.
```bash
$ twine upload dist/*
```
