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
const SEGMENT_STEP = FULL_CIRCLE / SEGMENT_COUNT;
const SEGMENT_LENGTH = SEGMENT_STEP - SEGMENT_GAP;
const NORTH_ANGLE = Math.PI / 2;
const STAR_GLOW_DURATION_MS = 1_800;
const STAR_GLOW_BASE_OPACITY = 0.32;
const STAR_GOLD = "#f4c76a";
const STAR_GLOW = "#ffab57";
const START_HEADING_DEGREES = 180;
const QUESTION_HEADINGS = [
  -90,
  110,
  -105,
  75,
  -60,
  60,
  -40,
  32,
  -18,
  8,
  -2,
] as const;
const COMPLETED = "#5576d9";
const MUTED = "#8290a6";
const INK = "#14223b";
const PAPER = "#f7faf9";

export function compassMotionEnabled(reducedMotion: boolean) {
  return !reducedMotion;
}

export interface StarGlowFrame {
  active: boolean;
  opacity: number;
  scale: number;
}

export function getStarGlowFrame(elapsedMs: number): StarGlowFrame {
  const progress =
    (Math.max(0, elapsedMs) % STAR_GLOW_DURATION_MS) /
    STAR_GLOW_DURATION_MS;
  const intensity = (1 - Math.cos(progress * FULL_CIRCLE)) / 2;
  return {
    active: true,
    opacity: STAR_GLOW_BASE_OPACITY + intensity * 0.32,
    scale: 1 + intensity * 0.18,
  };
}

export interface CompassMotionTuning {
  damping: number;
  kickDegreesPerSecond: number;
  stiffness: number;
}

export function getCompassMotionTuning(
  currentQuestionIndex: number | null,
  settled: boolean,
  complete = false,
): CompassMotionTuning {
  if (currentQuestionIndex === null || complete) {
    return { damping: 10, kickDegreesPerSecond: 0, stiffness: 72 };
  }
  const tuning = currentQuestionIndex < 4
    ? { damping: 3.4, kickDegreesPerSecond: 52, stiffness: 38 }
    : currentQuestionIndex < 8
      ? { damping: 5.8, kickDegreesPerSecond: 30, stiffness: 50 }
      : { damping: 8.4, kickDegreesPerSecond: 14, stiffness: 64 };
  return settled
    ? { ...tuning, kickDegreesPerSecond: tuning.kickDegreesPerSecond * 0.42 }
    : tuning;
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
  motion: CompassMotionTuning;
  mostSelected: boolean;
  settled: boolean;
}

interface SegmentVisual {
  mesh: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
}

export interface CompassSegmentLayout {
  svgDashLength: number;
  svgRotationDegrees: number;
  thetaLength: number;
  thetaStart: number;
}

export function getCompassSegmentLayout(
  index: number,
): CompassSegmentLayout {
  return {
    svgDashLength: SEGMENT_LENGTH / FULL_CIRCLE * 100,
    svgRotationDegrees:
      -90 + SEGMENT_GAP * 90 / Math.PI + index * 360 / SEGMENT_COUNT,
    thetaLength: SEGMENT_LENGTH,
    thetaStart:
      NORTH_ANGLE - SEGMENT_GAP / 2 - SEGMENT_LENGTH - index * SEGMENT_STEP,
  };
}

interface CompassScene {
  camera: THREE.PerspectiveCamera;
  complete: boolean;
  guideMaterial: THREE.LineBasicMaterial;
  instrument: THREE.Group;
  needle: THREE.Group;
  needleHeading: number;
  needleKickDirection: number;
  needleTarget: number;
  needleTuning: CompassMotionTuning;
  needleVelocity: number;
  northMaterial: THREE.MeshBasicMaterial;
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  segments: SegmentVisual[];
  southMaterial: THREE.MeshBasicMaterial;
  star: THREE.Mesh<THREE.ShapeGeometry, THREE.MeshBasicMaterial>;
  starGlowStartedAt: number | null;
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
  const safeQuestionIndex = currentQuestionIndex === null
    ? null
    : Math.min(SEGMENT_COUNT - 1, Math.max(0, currentQuestionIndex));
  const settled = complete || (mostSelected && leastSelected);
  const headingDegrees = complete
    ? 0
    : safeQuestionIndex === null
      ? START_HEADING_DEGREES
      : QUESTION_HEADINGS[safeQuestionIndex];

  return {
    calibration,
    complete,
    completedSegments,
    currentSegment: safeQuestionIndex,
    headingDegrees,
    leastSelected,
    motion: getCompassMotionTuning(safeQuestionIndex, settled, complete),
    mostSelected,
    settled,
  };
}

function applyVisualState(
  compass: CompassScene,
  state: CompassVisualState,
  reducedMotion: boolean,
) {
  const wasComplete = compass.complete;
  compass.complete = state.complete;
  compass.segments.forEach(({ mesh }, index) => {
    const completed = index < state.completedSegments;
    const current = index === state.currentSegment;
    const color = completed ? COMPLETED : MUTED;
    mesh.material.color.set(current ? PAPER : color);
    mesh.material.opacity = current ? 0.66 : completed ? 0.52 : 0.11;
  });

  const northActive = state.complete || state.mostSelected;
  const southActive = state.complete || state.leastSelected;
  compass.northMaterial.color.set(northActive ? "#2e8c82" : MUTED);
  compass.northMaterial.opacity = northActive ? 0.88 : 0.35;
  compass.southMaterial.color.set(southActive ? "#ff8a5b" : MUTED);
  compass.southMaterial.opacity = southActive ? 0.82 : 0.3;
  compass.needle.scale.setScalar(state.settled ? 1.025 : 1);

  compass.star.material.opacity = state.complete
    ? 1
    : 0.24 + state.calibration * 0.46;
  compass.star.material.color.set(state.complete ? STAR_GOLD : PAPER);
  compass.star.scale.setScalar(state.complete ? 1.18 : 1);
  compass.starHalo.material.color.set(state.complete ? STAR_GLOW : COMPLETED);
  compass.starHalo.material.opacity = state.complete
    ? STAR_GLOW_BASE_OPACITY
    : 0.035 + state.calibration * 0.08;
  compass.starHalo.scale.setScalar(1);
  if (!state.complete || reducedMotion) {
    compass.starGlowStartedAt = null;
  } else if (!wasComplete || compass.starGlowStartedAt === null) {
    compass.starGlowStartedAt = window.performance.now();
  }
  compass.guideMaterial.opacity = state.complete
    ? 0.34
    : 0.045 + state.calibration * 0.08;

  const rawTarget = THREE.MathUtils.degToRad(-state.headingDegrees);
  const nearestTurn = Math.round(
    (compass.needleHeading - rawTarget) / FULL_CIRCLE,
  );
  compass.needleTarget = rawTarget + nearestTurn * FULL_CIRCLE;
  compass.needleTuning = state.motion;
  if (reducedMotion) {
    compass.needleHeading = compass.needleTarget;
    compass.needleVelocity = 0;
    compass.needle.rotation.z = compass.needleTarget;
  }
}

function createNeedleGeometry(
  tipY: number,
  shoulderY: number,
  halfWidth: number,
  neckWidth: number,
) {
  const shape = new THREE.Shape();
  shape.moveTo(-neckWidth, 0);
  shape.lineTo(-halfWidth, shoulderY);
  shape.lineTo(0, tipY);
  shape.lineTo(halfWidth, shoulderY);
  shape.lineTo(neckWidth, 0);
  shape.closePath();
  return new THREE.ShapeGeometry(shape);
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

  const segments = Array.from({ length: SEGMENT_COUNT }, (_, index) => {
    const layout = getCompassSegmentLayout(index);
    const geometry = new THREE.RingGeometry(
      1.55,
      1.69,
      24,
      1,
      layout.thetaStart,
      layout.thetaLength,
    );
    const material = new THREE.MeshBasicMaterial({
      color: MUTED,
      depthWrite: false,
      opacity: 0.11,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    instrument.add(mesh);
    return { mesh };
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
  needle.rotation.z = THREE.MathUtils.degToRad(-START_HEADING_DEGREES);
  instrument.add(needle);

  const northMaterial = new THREE.MeshBasicMaterial({
    color: MUTED,
    depthWrite: false,
    opacity: 0.35,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const northNeedle = new THREE.Mesh(
    createNeedleGeometry(1.31, 0.18, 0.055, 0.026),
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
    createNeedleGeometry(-1.01, -0.16, 0.048, 0.023),
    southMaterial,
  );
  needle.add(southNeedle);

  const center = new THREE.Mesh(
    new THREE.CircleGeometry(0.13, 32),
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
    complete: false,
    guideMaterial,
    instrument,
    needle,
    needleHeading: THREE.MathUtils.degToRad(-START_HEADING_DEGREES),
    needleKickDirection: 1,
    needleTarget: THREE.MathUtils.degToRad(-START_HEADING_DEGREES),
    needleTuning: getCompassMotionTuning(null, false),
    needleVelocity: 0,
    northMaterial,
    renderer,
    scene,
    segments,
    southMaterial,
    star,
    starGlowStartedAt: null,
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
  const lastAppliedStateRef = useRef<CompassVisualState | null>(null);
  const startMotionRef = useRef<((addKick?: boolean) => void) | null>(null);
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
      compass.needleVelocity += displacement
        * compass.needleTuning.stiffness
        * delta;
      compass.needleVelocity *= Math.exp(
        -compass.needleTuning.damping * delta,
      );
      compass.needleHeading += compass.needleVelocity * delta;
      compass.needle.rotation.z = compass.needleHeading;

      let starGlowActive = false;
      if (compass.starGlowStartedAt !== null) {
        const glow = getStarGlowFrame(time - compass.starGlowStartedAt);
        compass.starHalo.material.opacity = glow.opacity;
        compass.starHalo.scale.setScalar(glow.scale);
        compass.star.scale.setScalar(1.18 + (glow.scale - 1) * 0.4);
        starGlowActive = glow.active;
      }
      renderScene();

      if (
        Math.abs(displacement) > 0.00035
        || Math.abs(compass.needleVelocity) > 0.00035
        || starGlowActive
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

    const startMotion = (addKick = false) => {
      window.cancelAnimationFrame(frame);
      previousTime = 0;
      if (!compassMotionEnabled(reducedMotionRef.current)) {
        compass.needleHeading = compass.needleTarget;
        compass.needleVelocity = 0;
        compass.needle.rotation.z = compass.needleTarget;
        renderScene();
        return;
      }
      if (addKick && compass.needleTuning.kickDegreesPerSecond > 0) {
        const displacement = compass.needleTarget - compass.needleHeading;
        if (Math.abs(displacement) > 0.00035) {
          compass.needleKickDirection = Math.sign(displacement);
        } else {
          compass.needleKickDirection *= -1;
        }
        compass.needleVelocity += compass.needleKickDirection
          * THREE.MathUtils.degToRad(
            compass.needleTuning.kickDegreesPerSecond,
          );
      }
      frame = window.requestAnimationFrame(renderMotion);
    };

    const handleMotionChange = (event: MediaQueryListEvent) => {
      reducedMotionRef.current = event.matches;
      applyVisualState(compass, visualStateRef.current, event.matches);
      startMotion(false);
    };
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      window.cancelAnimationFrame(frame);
      setWebglReady(false);
    };

    applyVisualState(compass, visualStateRef.current, motionQuery.matches);
    lastAppliedStateRef.current = visualStateRef.current;
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
      lastAppliedStateRef.current = null;
      disposeScene(compass);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const compass = sceneRef.current;
    if (!compass || lastAppliedStateRef.current === visualState) return;
    applyVisualState(compass, visualState, reducedMotionRef.current);
    lastAppliedStateRef.current = visualState;
    if (reducedMotionRef.current) {
      compass.renderer.render(compass.scene, compass.camera);
    } else {
      startMotionRef.current?.(true);
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
          const layout = getCompassSegmentLayout(index);
          return (
            <circle
              className={`living-compass__segment${completed ? " living-compass__segment--complete" : ""}${current ? " living-compass__segment--current" : ""}`}
              cx="50"
              cy="52"
              key={index}
              pathLength="100"
              r="40"
              strokeDasharray={`${layout.svgDashLength} ${100 - layout.svgDashLength}`}
              style={{
                "--segment-angle": `${layout.svgRotationDegrees}deg`,
              } as CSSProperties}
            />
          );
        })}
        <g
          className={`living-compass__needle${visualState.settled ? " living-compass__needle--settled" : ""}`}
          style={{ "--needle-angle": `${visualState.headingDegrees}deg` } as CSSProperties}
        >
          <path
            className={`living-compass__needle-half living-compass__needle-half--north${northActive ? " living-compass__needle-half--active" : ""}`}
            d="M49.35 52 L48.7 47.8 L50 18.5 L51.3 47.8 L50.65 52 Z"
          />
          <path
            className={`living-compass__needle-half living-compass__needle-half--south${southActive ? " living-compass__needle-half--active" : ""}`}
            d="M49.4 52 L48.85 56 L50 81 L51.15 56 L50.6 52 Z"
          />
        </g>
        <circle className="living-compass__center" cx="50" cy="52" r="3.3" />
      </svg>
    </div>
  );
}
