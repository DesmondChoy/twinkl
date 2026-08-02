import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import * as THREE from "three";

const SEGMENT_COUNT = 11;
const FULL_CIRCLE = Math.PI * 2;
const SEGMENT_GAP = 0.065;
const MAX_HEADING_DEGREES = 38;
const COMPLETED = "#5576d9";
const MUTED = "#8290a6";
const INK = "#14223b";
const PAPER = "#f7faf9";

export function compassMotionEnabled(reducedMotion: boolean) {
  return !reducedMotion;
}

export interface LivingCompassProps {
  currentQuestionIndex: number | null;
  leastSelected: boolean;
  milestone: number;
  mostSelected: boolean;
}

interface CompassVisualState {
  calibration: number;
  complete: boolean;
  completedSegments: number;
  currentSegment: number | null;
  headingDegrees: number;
  leastSelected: boolean;
  mostSelected: boolean;
  settled: boolean;
}

interface SegmentVisual {
  glow: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  mesh: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
}

interface NeedleTipVisual {
  halo: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>;
  point: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
}

interface CompassScene {
  camera: THREE.PerspectiveCamera;
  guideMaterial: THREE.LineBasicMaterial;
  instrument: THREE.Group;
  needle: THREE.Group;
  needleHeading: number;
  needleTarget: number;
  needleVelocity: number;
  northMaterial: THREE.MeshBasicMaterial;
  northTip: NeedleTipVisual;
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  segments: SegmentVisual[];
  shaftMaterial: THREE.LineBasicMaterial;
  southMaterial: THREE.MeshBasicMaterial;
  southTip: NeedleTipVisual;
  star: THREE.Mesh<THREE.ShapeGeometry, THREE.MeshBasicMaterial>;
  starHalo: THREE.Mesh<THREE.CircleGeometry, THREE.MeshBasicMaterial>;
}

function getVisualState({
  currentQuestionIndex,
  leastSelected,
  milestone,
  mostSelected,
}: LivingCompassProps): CompassVisualState {
  const complete = currentQuestionIndex === null && milestone > SEGMENT_COUNT;
  const completedSegments = complete
    ? SEGMENT_COUNT
    : Math.min(SEGMENT_COUNT, Math.max(0, milestone - 1));
  const calibration = completedSegments / SEGMENT_COUNT;
  const choiceCount = Number(mostSelected) + Number(leastSelected);
  const seekOffset = currentQuestionIndex === null
    ? 0
    : choiceCount === 0
      ? 3
      : choiceCount === 1
        ? 1.4
        : 0;
  const headingDegrees = complete
    ? 0
    : MAX_HEADING_DEGREES * (1 - calibration) + seekOffset;

  return {
    calibration,
    complete,
    completedSegments,
    currentSegment:
      currentQuestionIndex === null
        ? null
        : Math.min(SEGMENT_COUNT - 1, Math.max(0, currentQuestionIndex)),
    headingDegrees,
    leastSelected,
    mostSelected,
    settled: complete || (mostSelected && leastSelected),
  };
}

function applyTipState(
  tip: NeedleTipVisual,
  active: boolean,
  settled: boolean,
  color: string,
) {
  tip.point.material.color.set(active ? color : MUTED);
  tip.point.material.opacity = active ? 0.98 : 0.28;
  tip.point.scale.setScalar(active ? (settled ? 1.16 : 1.07) : 1);
  tip.halo.material.color.set(color);
  tip.halo.material.opacity = active ? (settled ? 0.42 : 0.28) : 0.04;
  tip.halo.scale.setScalar(active ? (settled ? 1.14 : 1.04) : 1);
}

function applyVisualState(
  compass: CompassScene,
  state: CompassVisualState,
  reducedMotion: boolean,
) {
  compass.segments.forEach(({ glow, mesh }, index) => {
    const completed = index < state.completedSegments;
    const current = index === state.currentSegment;
    const color = completed ? COMPLETED : MUTED;
    mesh.material.color.set(current ? PAPER : color);
    mesh.material.opacity = current ? 0.56 : completed ? 0.52 : 0.11;
    glow.visible = current;
    glow.material.color.set("#ff8a5b");
    glow.material.opacity = current ? 0.11 : 0;
  });

  const northActive = state.complete || state.mostSelected;
  const southActive = state.complete || state.leastSelected;
  compass.northMaterial.color.set(northActive ? "#2e8c82" : MUTED);
  compass.northMaterial.opacity = northActive ? 0.88 : 0.35;
  compass.southMaterial.color.set(southActive ? "#ff8a5b" : MUTED);
  compass.southMaterial.opacity = southActive ? 0.82 : 0.3;
  compass.shaftMaterial.opacity = state.settled ? 0.42 : 0.2;
  compass.needle.scale.setScalar(state.settled ? 1.025 : 1);
  applyTipState(compass.northTip, northActive, state.settled, "#2e8c82");
  applyTipState(compass.southTip, southActive, state.settled, "#ff8a5b");

  compass.star.material.opacity = state.complete
    ? 1
    : 0.24 + state.calibration * 0.46;
  compass.star.scale.setScalar(state.complete ? 1.18 : 1);
  compass.starHalo.material.opacity = state.complete
    ? 0.26
    : 0.035 + state.calibration * 0.08;
  compass.guideMaterial.opacity = state.complete
    ? 0.34
    : 0.045 + state.calibration * 0.08;

  compass.needleTarget = THREE.MathUtils.degToRad(-state.headingDegrees);
  if (reducedMotion) {
    compass.needleHeading = compass.needleTarget;
    compass.needleVelocity = 0;
    compass.needle.rotation.z = compass.needleTarget;
  }
}

function createNeedleGeometry(y: number, halfWidth: number) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([
      -halfWidth, 0, 0,
      0, y, 0,
      halfWidth, 0, 0,
    ], 3),
  );
  return geometry;
}

function createStarGeometry(outerRadius: number, innerRadius: number) {
  const shape = new THREE.Shape();
  Array.from({ length: 8 }, (_, index) => {
    const radius = index % 2 === 0 ? outerRadius : innerRadius;
    const angle = Math.PI / 2 + index * Math.PI / 4;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    if (index === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  });
  shape.closePath();
  return new THREE.ShapeGeometry(shape);
}

function buildScene(canvas: HTMLCanvasElement): CompassScene {
  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    canvas,
    powerPreference: "low-power",
  });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 20);
  camera.position.set(0, 0.02, 7.1);

  const instrument = new THREE.Group();
  instrument.rotation.x = -0.12;
  instrument.rotation.y = 0.05;
  scene.add(instrument);

  const segmentLength = FULL_CIRCLE / SEGMENT_COUNT - SEGMENT_GAP;
  const segments = Array.from({ length: SEGMENT_COUNT }, (_, index) => {
    const start = index * (FULL_CIRCLE / SEGMENT_COUNT) + SEGMENT_GAP / 2;
    const geometry = new THREE.RingGeometry(
      1.55,
      1.69,
      24,
      1,
      start,
      segmentLength,
    );
    const material = new THREE.MeshBasicMaterial({
      color: MUTED,
      depthWrite: false,
      opacity: 0.11,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const glowMaterial = new THREE.MeshBasicMaterial({
      blending: THREE.AdditiveBlending,
      color: PAPER,
      depthWrite: false,
      opacity: 0,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    const glow = new THREE.Mesh(geometry, glowMaterial);
    glow.scale.setScalar(1.035);
    glow.position.z = -0.015;
    glow.visible = false;
    instrument.add(glow, mesh);
    return { glow, mesh };
  });

  [1.12, 1.36].forEach((radius, index) => {
    const guide = new THREE.Mesh(
      new THREE.RingGeometry(radius, radius + 0.007, 64),
      new THREE.MeshBasicMaterial({
        color: index === 0 ? "#5576d9" : PAPER,
        opacity: index === 0 ? 0.12 : 0.075,
        side: THREE.DoubleSide,
        transparent: true,
      }),
    );
    guide.position.z = -0.025;
    instrument.add(guide);
  });

  const guideGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.2, 0.015),
    new THREE.Vector3(0, 1.79, 0.015),
  ]);
  const guideMaterial = new THREE.LineBasicMaterial({
    color: PAPER,
    opacity: 0.045,
    transparent: true,
  });
  instrument.add(new THREE.Line(guideGeometry, guideMaterial));

  const starHalo = new THREE.Mesh(
    new THREE.CircleGeometry(0.34, 32),
    new THREE.MeshBasicMaterial({
      blending: THREE.AdditiveBlending,
      color: COMPLETED,
      depthWrite: false,
      opacity: 0.035,
      transparent: true,
    }),
  );
  starHalo.position.set(0, 1.96, 0.02);
  instrument.add(starHalo);

  const star = new THREE.Mesh(
    createStarGeometry(0.21, 0.062),
    new THREE.MeshBasicMaterial({
      color: PAPER,
      depthWrite: false,
      opacity: 0.24,
      side: THREE.DoubleSide,
      transparent: true,
    }),
  );
  star.position.set(0, 1.96, 0.04);
  instrument.add(star);

  const needle = new THREE.Group();
  needle.position.z = 0.07;
  needle.rotation.z = THREE.MathUtils.degToRad(-MAX_HEADING_DEGREES);
  instrument.add(needle);

  const northMaterial = new THREE.MeshBasicMaterial({
    color: MUTED,
    depthWrite: false,
    opacity: 0.35,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const northNeedle = new THREE.Mesh(
    createNeedleGeometry(1.21, 0.115),
    northMaterial,
  );
  needle.add(northNeedle);

  const southMaterial = new THREE.MeshBasicMaterial({
    color: MUTED,
    depthWrite: false,
    opacity: 0.3,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const southNeedle = new THREE.Mesh(
    createNeedleGeometry(-0.92, 0.1),
    southMaterial,
  );
  needle.add(southNeedle);

  const shaftGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 1.23, 0.01),
    new THREE.Vector3(0, -0.94, 0.01),
  ]);
  const shaftMaterial = new THREE.LineBasicMaterial({
    color: PAPER,
    opacity: 0.2,
    transparent: true,
  });
  needle.add(new THREE.Line(shaftGeometry, shaftMaterial));

  const createTip = (y: number, color: string): NeedleTipVisual => {
    const point = new THREE.Mesh(
      new THREE.SphereGeometry(0.072, 16, 12),
      new THREE.MeshBasicMaterial({
        color: MUTED,
        depthWrite: false,
        opacity: 0.28,
        transparent: true,
      }),
    );
    point.position.set(0, y, 0.035);
    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.145, 0.011, 8, 32),
      new THREE.MeshBasicMaterial({
        blending: THREE.AdditiveBlending,
        color,
        depthWrite: false,
        opacity: 0.04,
        transparent: true,
      }),
    );
    halo.position.set(0, y, 0.025);
    needle.add(halo, point);
    return { halo, point };
  };

  const northTip = createTip(1.21, "#2e8c82");
  const southTip = createTip(-0.92, "#ff8a5b");

  const center = new THREE.Mesh(
    new THREE.CircleGeometry(0.17, 32),
    new THREE.MeshBasicMaterial({
      color: INK,
      opacity: 0.98,
      side: THREE.DoubleSide,
      transparent: true,
    }),
  );
  center.position.z = 0.11;
  instrument.add(center);

  return {
    camera,
    guideMaterial,
    instrument,
    needle,
    needleHeading: THREE.MathUtils.degToRad(-MAX_HEADING_DEGREES),
    needleTarget: THREE.MathUtils.degToRad(-MAX_HEADING_DEGREES),
    needleVelocity: 0,
    northMaterial,
    northTip,
    renderer,
    scene,
    segments,
    shaftMaterial,
    southMaterial,
    southTip,
    star,
    starHalo,
  };
}

function disposeScene(compass: CompassScene) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  compass.scene.traverse((object) => {
    if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
      geometries.add(object.geometry);
      const objectMaterials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      objectMaterials.forEach((material) => materials.add(material));
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
  compass.renderer.dispose();
}

export default function LivingCompass(props: LivingCompassProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<CompassScene | null>(null);
  const startMotionRef = useRef<(() => void) | null>(null);
  const reducedMotionRef = useRef(false);
  const visualState = useMemo(() => getVisualState(props), [
    props.currentQuestionIndex,
    props.leastSelected,
    props.milestone,
    props.mostSelected,
  ]);
  const visualStateRef = useRef(visualState);
  const [webglReady, setWebglReady] = useState(false);

  visualStateRef.current = visualState;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    if (/jsdom/i.test(navigator.userAgent)) return;

    let compass: CompassScene;
    try {
      compass = buildScene(canvas);
    } catch {
      setWebglReady(false);
      return;
    }

    sceneRef.current = compass;
    let disposed = false;
    let frame = 0;
    let previousTime = 0;
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = motionQuery.matches;

    const renderScene = () => {
      compass.renderer.render(compass.scene, compass.camera);
    };

    const resize = () => {
      const bounds = container.getBoundingClientRect();
      const size = Math.max(1, Math.min(bounds.width, bounds.height));
      compass.renderer.setSize(size, size, false);
      compass.camera.aspect = 1;
      compass.camera.updateProjectionMatrix();
      renderScene();
    };

    const renderMotion = (time: number) => {
      if (
        disposed
        || !compassMotionEnabled(reducedMotionRef.current)
      ) return;
      const delta = previousTime === 0
        ? 0
        : Math.min((time - previousTime) / 1_000, 0.04);
      previousTime = time;
      const displacement = compass.needleTarget - compass.needleHeading;
      compass.needleVelocity += displacement * 82 * delta;
      compass.needleVelocity *= Math.exp(-12 * delta);
      compass.needleHeading += compass.needleVelocity * delta;
      compass.needle.rotation.z = compass.needleHeading;
      renderScene();

      if (
        Math.abs(displacement) > 0.00035
        || Math.abs(compass.needleVelocity) > 0.00035
      ) {
        frame = window.requestAnimationFrame(renderMotion);
      } else {
        compass.needleHeading = compass.needleTarget;
        compass.needleVelocity = 0;
        compass.needle.rotation.z = compass.needleTarget;
        renderScene();
        frame = 0;
      }
    };

    const startMotion = () => {
      window.cancelAnimationFrame(frame);
      previousTime = 0;
      if (!compassMotionEnabled(reducedMotionRef.current)) {
        compass.needleHeading = compass.needleTarget;
        compass.needleVelocity = 0;
        compass.needle.rotation.z = compass.needleTarget;
        renderScene();
        return;
      }
      frame = window.requestAnimationFrame(renderMotion);
    };

    const handleMotionChange = (event: MediaQueryListEvent) => {
      reducedMotionRef.current = event.matches;
      applyVisualState(compass, visualStateRef.current, event.matches);
      startMotion();
    };
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      window.cancelAnimationFrame(frame);
      setWebglReady(false);
    };

    applyVisualState(compass, visualStateRef.current, motionQuery.matches);
    startMotionRef.current = startMotion;
    resize();
    startMotion();
    setWebglReady(true);
    canvas.addEventListener("webglcontextlost", handleContextLost);
    motionQuery.addEventListener("change", handleMotionChange);

    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(resize);
    if (resizeObserver) {
      resizeObserver.observe(container);
    } else {
      window.addEventListener("resize", resize);
    }

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      motionQuery.removeEventListener("change", handleMotionChange);
      resizeObserver?.disconnect();
      if (!resizeObserver) window.removeEventListener("resize", resize);
      startMotionRef.current = null;
      disposeScene(compass);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const compass = sceneRef.current;
    if (!compass) return;
    applyVisualState(compass, visualState, reducedMotionRef.current);
    if (reducedMotionRef.current) {
      compass.renderer.render(compass.scene, compass.camera);
    } else {
      startMotionRef.current?.();
    }
  }, [visualState]);

  const northActive = visualState.complete || visualState.mostSelected;
  const southActive = visualState.complete || visualState.leastSelected;

  return (
    <div
      className={`compass living-compass${webglReady ? " living-compass--webgl" : " living-compass--fallback"}`}
      ref={containerRef}
      aria-hidden="true"
    >
      <canvas
        className="living-compass__canvas"
        ref={canvasRef}
        aria-hidden="true"
      />
      <svg
        className="living-compass__fallback"
        viewBox="0 0 100 100"
        aria-hidden="true"
      >
        <line
          className="living-compass__north-guide"
          x1="50"
          x2="50"
          y1="12"
          y2="22"
        />
        <path
          className={`living-compass__north-star${visualState.complete ? " living-compass__north-star--complete" : ""}`}
          d="M50 1.5 L51.7 6.1 L56.5 8 L51.7 9.9 L50 14.5 L48.3 9.9 L43.5 8 L48.3 6.1 Z"
          style={{ "--star-progress": visualState.calibration } as CSSProperties}
        />
        <circle className="living-compass__guide" cx="50" cy="52" r="31" />
        <circle className="living-compass__guide" cx="50" cy="52" r="36" />
        {Array.from({ length: SEGMENT_COUNT }, (_, index) => {
          const completed = index < visualState.completedSegments;
          const current = index === visualState.currentSegment;
          return (
            <circle
              className={`living-compass__segment${completed ? " living-compass__segment--complete" : ""}${current ? " living-compass__segment--current" : ""}`}
              cx="50"
              cy="52"
              key={index}
              pathLength="100"
              r="40"
              style={{ "--segment-index": index } as CSSProperties}
            />
          );
        })}
        <g
          className={`living-compass__needle${visualState.settled ? " living-compass__needle--settled" : ""}`}
          style={{ "--needle-angle": `${visualState.headingDegrees}deg` } as CSSProperties}
        >
          <line
            className="living-compass__needle-line"
            x1="50"
            x2="50"
            y1="22"
            y2="78"
          />
          <path
            className={`living-compass__needle-half living-compass__needle-half--north${northActive ? " living-compass__needle-half--active" : ""}`}
            d="M47.8 52 L50 22 L52.2 52 Z"
          />
          <path
            className={`living-compass__needle-half living-compass__needle-half--south${southActive ? " living-compass__needle-half--active" : ""}`}
            d="M48.1 52 L50 78 L51.9 52 Z"
          />
          <circle
            className={`living-compass__anchor living-compass__anchor--most${northActive ? " living-compass__anchor--selected" : ""}${visualState.settled ? " living-compass__anchor--settled" : ""}`}
            cx="50"
            cy="22"
            r="1.8"
          />
          <circle
            className={`living-compass__anchor living-compass__anchor--least${southActive ? " living-compass__anchor--selected" : ""}${visualState.settled ? " living-compass__anchor--settled" : ""}`}
            cx="50"
            cy="78"
            r="1.8"
          />
        </g>
        <circle className="living-compass__center" cx="50" cy="52" r="4.4" />
      </svg>
    </div>
  );
}
