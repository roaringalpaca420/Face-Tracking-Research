# Face Tracker Avatar Demo

MediaPipe + Three.js face-tracking avatar. Works in the browser on desktop and phone.

---

## Test on your phone (GitHub Pages)

1. **Push this folder to a GitHub repo** (if you haven’t already):
   ```bash
   git init
   git add index.html main.js .nojekyll README.md
   git commit -m "Face tracker avatar web app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Turn on GitHub Pages**  
   On GitHub: **Settings → Pages → Build and deployment**  
   - Source: **Deploy from a branch**  
   - Branch: **main** (or **master**)  
   - Folder: **/ (root)**  
   - Save.

3. **Open the site on your phone**  
   After a minute or two, the site will be at:
   ```text
   https://YOUR_USERNAME.github.io/YOUR_REPO/
   ```
   Open that URL on your phone, allow camera access when asked, and use the front camera to drive the avatar.

**Note:** Camera only works over **HTTPS**. GitHub Pages gives you HTTPS, so it works on mobile.

---

## Run locally (desktop)

1. **Start a local server** (needed for camera and ES modules):
   ```bash
   npx serve .
   ```
   Or: `python -m http.server 8000`

2. **Open in a browser**  
   - With `npx serve .`: **http://localhost:3000**  
   - With Python: **http://localhost:8000**

3. **Allow camera access** when prompted.

---

## Files

| File | Purpose |
|------|--------|
| **index.html** | Page with hidden `<video>`, mobile-friendly layout |
| **main.js** | Demo logic (webcam → MediaPipe → Three.js avatar), with Safari/iOS fallback |
| **.nojekyll** | Tells GitHub Pages not to use Jekyll |
| **MediaPipe Face Virtual Avatar Demo.txt** | Original TypeScript reference |

## Change the avatar

Edit `main.js` and update the URL in the `Avatar` constructor:

```js
const avatar = new Avatar(
  "https://your-url-here/your_model.glb",
  scene.scene
);
```

Use a GLB with blendshapes for best facial motion.
