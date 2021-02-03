function dtypeToType(dtype) {
    switch(dtype) {
      case 'uint8':
        return Uint8Array;
      case 'uint16':
        return Uint16Array;
      case 'uint32':
        return Uint32Array;
      case 'int8':
        return Int8Array;
      case 'int16':
        return Int16Array;
      case 'int32':
        return Int32Array;
      case 'float':
      case 'float32':
        return Float32Array;
      case 'double':
      case 'float64':
        return Float64Array;
      case 'uint8_clamped':
        return Uint8ClampedArray;
      case 'generic':
      case 'buffer':
      case 'data':
      case 'dataview':
        return ArrayBuffer;
      case 'array':
        return Array;
    }
  }
  
function zeros(shape, dtype) {
    dtype = dtype || 'float64';
    var sz = 1;
    for(var i=0; i<shape.length; ++i) {
      sz *= shape[i];
    }
    return ndarray(new (dtypeToType(dtype))(sz), shape);
  }

/**
 * For all below functions the parameter mu defines where to estimate the value on the interpolated
 * line, it is 0 at y1 and 1 at y2. Furthermore all y points are assumed to be equally spaced on the
 * x-axis.
 */

// Connects the points with a straight line
const linearInterpolate = (mu, y1, y2) => {
    const y = y1 * (1 - mu) + y2 * mu;
  
    return y;
  };
  
  // Uses cosine for smoother transitions between points
  const cosineInterpolate = (mu, y1, y2) => {
    const mu2 = (1 - Math.cos(mu * Math.PI)) / 2;
    const y = y1 * (1 - mu2) + y2 * mu2;
  
    return y;
  };
  
  // Cubic interpolation for a smooth line through all four points
  const cubicInterpolate = (mu, y0, y1, y2, y3) => {
    const mu2 = mu * mu;
    const a0 = y3 - y2 - y0 + y1;
    const a1 = y0 - y1 - a0;
    const a2 = y2 - y0;
    const a3 = y1;
    const y = a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
  
    return Math.max(0, Math.min(y, 255));
  };
  
  // Catmull-Rom cubic interpolation for a slightly smoother line through all four points
  const cattmullRomInterpolate = (mu, y0, y1, y2, y3) => {
    const mu2 = mu * mu;
    const a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    const a1 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3;
    const a2 = -0.5 * y0 + 0.5 * y2;
    const a3 = y1;
    const y = a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
  
    return Math.max(0, Math.min(y, 255));
  };
  
  /**
   * Hermite interpolation with bias and tension controls. For tension: 1 is high, 0 normal, -1 is
   * low. For bias: 0 is even, positive is towards first segment, negative towards the other.
   */
  
  const hermiteInterpolate = (mu, y0, y1, y2, y3, bias, tension) => {
    const mu2 = mu * mu;
    const mu3 = mu2 * mu;
  
    let m0 = ((y1 - y0) * (1 + bias) * (1 - tension)) / 2;
    let m1 = ((y2 - y1) * (1 + bias) * (1 - tension)) / 2;
    m0 += ((y2 - y1) * (1 - bias) * (1 - tension)) / 2;
    m1 += ((y3 - y2) * (1 - bias) * (1 - tension)) / 2;
  
    const a0 = 2 * mu3 - 3 * mu2 + 1;
    const a1 = mu3 - 2 * mu2 + mu;
    const a2 = mu3 - mu2;
    const a3 = -2 * mu3 + 3 * mu2;
  
    const y = a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2;
  
    return Math.max(0, Math.min(y, 255));
  };
  
  // Bezier interpolation through four points
  const bezierInterpolate = (mu, y0, y1, y2, y3) => {
    const cp1 = y1 + (y2 - y0) / 4;
    const cp2 = y2 - (y3 - y1) / 4;
    const nt = 1 - mu;
    const c0 = y1 * nt * nt * nt;
    const c1 = 3 * cp1 * nt * nt * mu;
    const c2 = 3 * cp2 * nt * mu * mu;
    const c3 = y2 * mu * mu * mu;
    const y = c0 + c1 + c2 + c3;
  
    return Math.max(0, Math.min(y, 255));
  };

  /**
 * Nearest neighbour interpolation
 */

const nearestNeighbour = ({
    original,
    originalWidth,
    originalHeight,
    originalX,
    originalY,
    colorIndex
  }) => {
    const neighbourX = Math.min(Math.floor(originalX), originalWidth - 1);
    const neighbourY = Math.min(Math.floor(originalY), originalHeight - 1);
    return original.get(neighbourX, neighbourY, colorIndex);
  };
  
  /**
   * Higher level function that returns a function that interpolates between 2 points in both
   * directions. Since there are multiple algorithms that can be used for interpolation, this
   * simplifies implementation a bit.
   */
  
  const generate4PointInterpolator = interpolator => ({
    original,
    originalWidth,
    originalHeight,
    originalX,
    originalY,
    colorIndex
  }) => {
    const x0 = Math.floor(originalX);
    const y0 = Math.floor(originalY);
    const x1 = Math.min(Math.ceil(originalX), originalWidth - 1);
    const y1 = Math.min(Math.ceil(originalY), originalHeight - 1);
  
    // If the target matches an existing coordinate there's no need to interpolate
    if (x0 === x1 && y0 === y1) {
      return original.get(x0, y0, colorIndex);
    }
  
    // The values of the coordinates surrounding the interpolation target
    const x0y0 = original.get(x0, y0, colorIndex);
    const x0y1 = original.get(x0, y1, colorIndex);
    const x1y1 = original.get(x1, y1, colorIndex);
    const x1y0 = original.get(x1, y0, colorIndex);
  
    // The x and y values to interpolate expressed as numbers between 0 and 1
    const xMu = originalX % 1;
    const yMu = originalY % 1;
  
    // Interpolate for y0 and y1 rows
    const r0 = interpolator(xMu, x0y0, x1y0);
    const r1 = interpolator(xMu, x0y1, x1y1);
  
    // If the target matches an existing x-coordinate there's no need to interpolate for x
    if (x0 === x1) {
      return interpolator(yMu, x0y0, x0y1);
    }
  
    // If the target matches an existing y-coordinate there's no need to interpolate for y
    if (y0 === y1) {
      return r0;
    }
  
    return interpolator(yMu, r0, r1);
  };
  
  /**
   * Higher level function that returns a function that interpolates between 4 points in both
   * directions. Since there are multiple algorithms that can be used for interpolation, this
   * simplifies implementation a bit.
   */
  
  const generate16PointInterpolator = interpolator => ({
    original,
    originalWidth,
    originalHeight,
    originalX,
    originalY,
    colorIndex,
    bias,
    tension
  }) => {
    const x0 = Math.floor(originalX) - 1;
    const y0 = Math.floor(originalY) - 1;
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const x2 = Math.min(Math.ceil(originalX), originalWidth - 1);
    const y2 = Math.min(Math.ceil(originalY), originalHeight - 1);
    const x3 = x2 + 1;
    const y3 = y2 + 1;
  
    // If the target matches an existing coordinate there's no need to interpolate
    if (x1 === x2 && y1 === y2) {
      return original.get(x1, y1, colorIndex);
    }
  
    // Boundary values for x and y
    const xMin = 0;
    const xMax = originalWidth - 1;
    const yMin = 0;
    const yMax = originalHeight - 1;
  
    // Get known values
    const x1y1 = original.get(x1, y1, colorIndex);
    const x2y1 = original.get(x2, y1, colorIndex);
    const x1y2 = original.get(x1, y2, colorIndex);
    const x2y2 = original.get(x2, y2, colorIndex);
  
    // Get or interpolate values to the left, right, above or below of known values
    const x0y1 = x0 >= xMin ? original.get(x0, y1, colorIndex) : linearInterpolate(-1, x1y1, x2y1);
    const x0y2 = x0 >= xMin ? original.get(x0, y2, colorIndex) : linearInterpolate(-1, x1y2, x2y2);
    const x1y0 = y0 >= yMin ? original.get(x1, y0, colorIndex) : linearInterpolate(-1, x1y1, x1y2);
    const x2y0 = y0 >= yMin ? original.get(x2, y0, colorIndex) : linearInterpolate(-1, x2y1, x2y2);
    const x3y1 = x3 <= xMax ? original.get(x3, y1, colorIndex) : linearInterpolate(2, x1y1, x2y1);
    const x3y2 = x3 <= xMax ? original.get(x3, y2, colorIndex) : linearInterpolate(2, x1y2, x2y2);
    const x1y3 = y3 <= yMax ? original.get(x1, y3, colorIndex) : linearInterpolate(2, x1y1, x1y2);
    const x2y3 = y3 <= yMax ? original.get(x2, y3, colorIndex) : linearInterpolate(2, x2y1, x2y2);
  
    // Get or interpolate corner values
    const x0y0 =
      x0 >= xMin && y0 >= yMin ? original.get(x0, y0, colorIndex) : linearInterpolate(-1, x1y0, x2y0);
    const x3y0 =
      x3 <= xMax && y0 >= yMin ? original.get(x3, y0, colorIndex) : linearInterpolate(2, x1y0, x2y0);
    const x0y3 =
      x0 >= xMin && y3 <= yMax ? original.get(x0, y3, colorIndex) : linearInterpolate(-1, x1y3, x2y3);
    const x3y3 =
      x3 <= xMax && y3 <= yMax ? original.get(x3, y3, colorIndex) : linearInterpolate(2, x1y3, x2y3);
  
    // The x and y values to interpolate expressed as numbers between 0 and 1
    const xMu = originalX % 1;
    const yMu = originalY % 1;
  
    const r0 = interpolator(xMu, x0y0, x1y0, x2y0, x3y0, bias, tension);
    const r1 = interpolator(xMu, x0y1, x1y1, x2y1, x3y1, bias, tension);
    const r2 = interpolator(xMu, x0y2, x1y2, x2y2, x3y2, bias, tension);
    const r3 = interpolator(xMu, x0y3, x1y3, x2y3, x3y3, bias, tension);
  
    return interpolator(yMu, r0, r1, r2, r3, bias, tension);
  };
  
const algorithms = {
    nearestNeighbour,
    bilinear: generate4PointInterpolator(linearInterpolate),
    cosine: generate4PointInterpolator(cosineInterpolate),
    bicubic: generate16PointInterpolator(cubicInterpolate),
    cattmullRom: generate16PointInterpolator(cattmullRomInterpolate),
    hermite: generate16PointInterpolator(hermiteInterpolate),
    bezier: generate16PointInterpolator(bezierInterpolate)
  };
  
  const resize_image = (original, options = {}) => {
    const originalWidth = original.shape[0];
    const originalHeight = original.shape[1];
  
    /* istanbul ignore next */
    const algorithm = options.algorithm || 'bilinear';
    const targetWidth = options.targetWidth || originalWidth;
    const targetHeight = options.targetHeight || originalHeight;
    const bias = options.bias || 0;
    const tension = options.tension || 0;
    const xScale = targetWidth / originalWidth;
    const yScale = targetHeight / originalHeight;
    const resizer = algorithms[algorithm];
  
    /* istanbul ignore next */
    if (targetWidth === originalWidth && targetHeight === originalHeight) {
      return original;
    }
  
    const result = zeros([targetWidth, targetHeight, 3]);
    const constants = { original, originalWidth, originalHeight, bias, tension };
  
    for (let yIndex = 0; yIndex < targetHeight; yIndex += 1) {
      for (let xIndex = 0; xIndex < targetWidth; xIndex += 1) {
        const originalX = xIndex / xScale;
        const originalY = yIndex / yScale;
  
        const red = Math.round(resizer({ ...constants, originalX, originalY, colorIndex: 0 }));
        const green = Math.round(resizer({ ...constants, originalX, originalY, colorIndex: 1 }));
        const blue = Math.round(resizer({ ...constants, originalX, originalY, colorIndex: 2 }));
  
        result.set(xIndex, yIndex, 0, red);
        result.set(xIndex, yIndex, 1, green);
        result.set(xIndex, yIndex, 2, blue);
      }
    }
  
    return result;
  };