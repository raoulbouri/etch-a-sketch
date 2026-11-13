// three.js view of the MuJoCo scene. Geometry comes from the compiled model, so what is drawn is
// what the physics uses. MuJoCo's world is z-up; the camera's up vector is set to match.
import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

const GEOM = { PLANE: 0, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

export class View {
  constructor(container, sim, screenCanvas, theme) {
    this.sim = sim; this.container = container;
    const renderer = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label', 'Simulated robot arm turning the knobs of an Etch A Sketch');
    renderer.domElement.setAttribute('role', 'img');

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.02, 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.18, -0.44, 0.74);       // same side as the free camera of scripts/make_video.py
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.target.set(0.34, 0.0, 0.17);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.3; this.controls.maxDistance = 2.5;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
    this.controls.update();

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a8a84, 1.1);
    this.hemi.position.set(0, 0, 1);
    this.scene.add(this.hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(0.5, -0.9, 1.6);
    sun.target.position.set(0.35, 0, 0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -0.7, right: 0.7, top: 0.7, bottom: -0.7, near: 0.2, far: 4 });
    sun.shadow.bias = -0.0004;
    this.scene.add(sun, sun.target);

    this.screenTex = new THREE.CanvasTexture(screenCanvas);
    this.screenTex.colorSpace = THREE.SRGBColorSpace;
    this.screenTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    this._build();
    this.setTheme(theme);
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  setTheme(theme) {
    const dark = theme === 'dark';
    this.scene.background = new THREE.Color(dark ? 0x101513 : 0xeef0ee);
    this.table.material.color.set(dark ? 0x2a302d : 0xdbd6cc);
    this.hemi.intensity = dark ? 0.8 : 1.1;
  }

  _meshGeometry(id) {
    const m = this.sim.m;
    const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
    const vert = m.mesh_vert, face = m.mesh_face;
    const pos = new Float32Array(fn * 9);              // unshared vertices: flat shading suits CAD meshes
    for (let f = 0; f < fn; f++) {
      for (let k = 0; k < 3; k++) {
        const v = va + face[3 * (fa + f) + k];
        pos.set([vert[3 * v], vert[3 * v + 1], vert[3 * v + 2]], 9 * f + 3 * k);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  }

  _build() {
    const { sim } = this, m = sim.m;
    const named = {};
    for (const n of ['toy_screen', 'marker0', 'marker1', 'marker2', 'marker3', 'table']) named[sim.id.geom(n)] = n;
    const loader = new THREE.TextureLoader();
    const meshCache = {};
    this.objects = [];
    const type = m.geom_type, size = m.geom_size, group = m.geom_group, matid = m.geom_matid, dataid = m.geom_dataid;
    const grgba = m.geom_rgba, mrgba = m.mat_rgba;
    for (let g = 0; g < m.ngeom; g++) {
      if (group[g] >= 3) continue;                     // collision geometry
      const [sx, sy, sz] = [size[3 * g], size[3 * g + 1], size[3 * g + 2]];
      const rgba = matid[g] >= 0 ? mrgba.subarray(4 * matid[g], 4 * matid[g] + 4) : grgba.subarray(4 * g, 4 * g + 4);
      let geo;
      switch (type[g]) {
        case GEOM.PLANE: geo = new THREE.PlaneGeometry(6, 6); break;
        case GEOM.BOX: geo = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz); break;
        case GEOM.SPHERE: geo = new THREE.SphereGeometry(sx, 24, 16); break;
        case GEOM.CYLINDER: geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 48).rotateX(Math.PI / 2); break;
        case GEOM.CAPSULE: geo = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16).rotateX(Math.PI / 2); break;
        case GEOM.MESH: geo = meshCache[dataid[g]] ??= this._meshGeometry(dataid[g]); break;
        default: continue;
      }
      const name = named[g];
      const base = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: 0.6, metalness: 0.05,
        transparent: rgba[3] < 1, opacity: rgba[3],
      });
      let material = base;
      if (name === 'toy_screen' || (name && name.startsWith('marker'))) {
        const map = name === 'toy_screen' ? this.screenTex
          : loader.load(new URL(`../scene/aruco_4x4_${name.slice(6)}.png`, import.meta.url).href);
        if (name !== 'toy_screen') { map.colorSpace = THREE.SRGBColorSpace; map.magFilter = THREE.NearestFilter; }
        const top = new THREE.MeshStandardMaterial({ map, roughness: name === 'toy_screen' ? 0.35 : 0.9 });
        base.color.set(name === 'toy_screen' ? 0xbcbcb6 : 0xffffff);
        material = [base, base, base, base, top, base];        // BoxGeometry face order: +x -x +y -y +z -z
      }
      const mesh = new THREE.Mesh(geo, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = type[g] !== GEOM.PLANE && !name;
      mesh.receiveShadow = true;
      if (name === 'table') this.table = mesh;
      this.scene.add(mesh);
      this.objects.push([g, mesh]);
    }
    this.sync();
  }

  /** Copy geom poses from the simulation. */
  sync() {
    const p = this.sim.d.geom_xpos, R = this.sim.d.geom_xmat;
    for (const [g, mesh] of this.objects) {
      const i = 9 * g, j = 3 * g;
      mesh.matrix.set(R[i], R[i + 1], R[i + 2], p[j], R[i + 3], R[i + 4], R[i + 5], p[j + 1],
        R[i + 6], R[i + 7], R[i + 8], p[j + 2], 0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  screenChanged() { this.screenTex.needsUpdate = true; }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.fov = w / h < 1 ? 52 : 40;             // narrow layouts need a wider view to keep the arm in frame
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
