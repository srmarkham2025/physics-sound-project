import _extends from "@babel/runtime/helpers/extends";
/**
 * THIS FILE IS AUTO-GENERATED
 * DON'T MAKE CHANGES HERE
 */
import { config } from './configReadonly.js';
import { createNode, createObjectNode, createOperatorNode, createParenthesisNode, createRelationalNode, createArrayNode, createBlockNode, createConditionalNode, createConstantNode, createRangeNode, createReviver, createChainClass, createFunctionAssignmentNode, createChain, createIndexNode, createAccessorNode, createAssignmentNode, createSymbolNode, createFunctionNode, createParse, createResolve, createSimplifyConstant, createCompile, createSimplifyCore, createEvaluate, createHelpClass, createParserClass, createSimplify, createSymbolicEqual, createLeafCount, createParser, createRationalize, createDerivative, createHelp, createMapSlicesTransform, createFilterTransform, createForEachTransform, createMapTransform, createOrTransform, createAndTransform, createConcatTransform, createMaxTransform, createPrintTransform, createSumTransform, createBitAndTransform, createMinTransform, createNullishTransform, createSubsetTransform, createBitOrTransform, createDiffTransform, createIndexTransform, createRangeTransform, createRowTransform, createColumnTransform, createCumSumTransform, createMeanTransform, createQuantileSeqTransform, createVarianceTransform, createStdTransform } from '../factoriesAny.js';
import { BigNumber, Complex, e, _false, fineStructure, Fraction, i, _Infinity, LN10, LOG10E, Matrix, _NaN, _null, phi, Range, ResultSet, SQRT1_2,
// eslint-disable-line camelcase
sackurTetrode, tau, _true, version, DenseMatrix, efimovFactor, LN2, pi, replacer, SQRT2, typed, weakMixingAngle, abs, acos, acot, acsc, addScalar, arg, asech, asinh, atan, atanh, bigint, bitNot, boolean, clone, combinations, complex, conj, cos, cot, csc, cube, equalScalar, erf, exp, expm1, filter, flatten, forEach, format, getMatrixDataType, hex, im, isInteger, isNegative, isPositive, isZero, LOG2E, lgamma, log10, log2, map, multiplyScalar, not, number, oct, pickRandom, print, random, re, sec, sign, sin, size, SparseMatrix, splitUnit, square, string, subtractScalar, tan, toBest, typeOf, acosh, acsch, asec, bignumber, combinationsWithRep, cosh, csch, dot, isNaN, isPrime, mapSlices, matrix, matrixFromFunction, multiply, ones, randomInt, reshape, sech, sinh, sparse, sqrt, squeeze, tanh, transpose, xgcd, zeros, acoth, asin, bin, concat, coth, ctranspose, diag, dotMultiply, equal, fraction, identity, isNumeric, kron, largerEq, leftShift, matrixFromRows, mode, nthRoot, numeric, prod, resize, rightArithShift, round, smaller, to, unaryMinus, unequal, xor, add, atan2, bitAnd, bitOr, bitXor, cbrt, compare, compareText, count, deepEqual, divideScalar, equalText, floor, gcd, hasNumericValue, hypot, ImmutableDenseMatrix, Index, larger, log, lsolve, matrixFromColumns, min, mod, nthRoots, nullish, or, partitionSelect, qr, rightLogShift, smallerEq, subset, subtract, trace, usolve, zpk2tf, catalan, compareNatural, composition, cross, det, diff, distance, dotDivide, FibonacciHeap, index, intersect, invmod, lcm, log1p, lsolveAll, max, range, row, setCartesian, setDistinct, setIsSubset, setPowerset, slu, sort, unaryPlus, usolveAll, and, ceil, column, cumsum, fix, setDifference, setMultiplicity, setSymDifference, Spa, sum, inv, lup, pinv, pow, setIntersect, setUnion, sqrtm, Unit, vacuumImpedance, wienDisplacement, atomicMass, bohrMagneton, boltzmann, conductanceQuantum, coulomb, createUnit, deuteronMass, dotPow, electricConstant, elementaryCharge, expm, faraday, fft, gamma, gravitationConstant, hartreeEnergy, ifft, klitzing, loschmidt, magneticConstant, molarMass, molarPlanckConstant, neutronMass, nuclearMagneton, planckCharge, planckLength, planckTemperature, protonMass, quantumOfCirculation, reducedPlanckConstant, rydberg, secondRadiation, speedOfLight, stefanBoltzmann, thomsonCrossSection, avogadro, bohrRadius, coulombConstant, divide, electronMass, factorial, firstRadiation, gravity, inverseConductanceQuantum, lusolve, magneticFluxQuantum, molarMassC12, multinomial, permutations, planckMass, polynomialRoot, setSize, solveODE, stirlingS2, unit, bellNumbers, eigs, fermiCoupling, gasConstant, kldivergence, mean, molarVolume, planckConstant, quantileSeq, variance, classicalElectronRadius, median, corr, freqz, mad, std, zeta, norm, rotationMatrix, planckTime, schur, rotate, sylvester, lyap } from './pureFunctionsAny.generated.js';
var math = {}; // NOT pure!
var mathWithTransform = {}; // NOT pure!
var classes = {}; // NOT pure!

export var Node = createNode({
  mathWithTransform
});
export var ObjectNode = createObjectNode({
  Node
});
export var OperatorNode = createOperatorNode({
  Node
});
export var ParenthesisNode = createParenthesisNode({
  Node
});
export var RelationalNode = createRelationalNode({
  Node
});
export var ArrayNode = createArrayNode({
  Node
});
export var BlockNode = createBlockNode({
  Node,
  ResultSet
});
export var ConditionalNode = createConditionalNode({
  Node
});
export var ConstantNode = createConstantNode({
  Node
});
export var RangeNode = createRangeNode({
  Node
});
export var reviver = createReviver({
  classes
});
export var Chain = createChainClass({
  math,
  typed
});
export var FunctionAssignmentNode = createFunctionAssignmentNode({
  Node,
  typed
});
export var chain = createChain({
  Chain,
  typed
});
export var IndexNode = createIndexNode({
  Node,
  size
});
export var AccessorNode = createAccessorNode({
  Node,
  subset
});
export var AssignmentNode = createAssignmentNode({
  matrix,
  Node,
  subset
});
export var SymbolNode = createSymbolNode({
  Unit,
  Node,
  math
});
export var FunctionNode = createFunctionNode({
  Node,
  SymbolNode,
  math
});
export var parse = createParse({
  AccessorNode,
  ArrayNode,
  AssignmentNode,
  BlockNode,
  ConditionalNode,
  ConstantNode,
  FunctionAssignmentNode,
  FunctionNode,
  IndexNode,
  ObjectNode,
  OperatorNode,
  ParenthesisNode,
  RangeNode,
  RelationalNode,
  SymbolNode,
  config,
  numeric,
  typed
});
export var resolve = createResolve({
  ConstantNode,
  FunctionNode,
  OperatorNode,
  ParenthesisNode,
  parse,
  typed
});
export var simplifyConstant = createSimplifyConstant({
  bignumber,
  fraction,
  AccessorNode,
  ArrayNode,
  ConstantNode,
  FunctionNode,
  IndexNode,
  ObjectNode,
  OperatorNode,
  SymbolNode,
  config,
  mathWithTransform,
  matrix,
  typed
});
export var compile = createCompile({
  parse,
  typed
});
export var simplifyCore = createSimplifyCore({
  AccessorNode,
  ArrayNode,
  ConstantNode,
  FunctionNode,
  IndexNode,
  ObjectNode,
  OperatorNode,
  ParenthesisNode,
  SymbolNode,
  add,
  divide,
  equal,
  isZero,
  multiply,
  parse,
  pow,
  subtract,
  typed
});
export var evaluate = createEvaluate({
  parse,
  typed
});
export var Help = createHelpClass({
  evaluate
});
export var Parser = createParserClass({
  evaluate,
  parse
});
export var simplify = createSimplify({
  AccessorNode,
  ArrayNode,
  ConstantNode,
  FunctionNode,
  IndexNode,
  ObjectNode,
  OperatorNode,
  ParenthesisNode,
  SymbolNode,
  equal,
  parse,
  replacer,
  resolve,
  simplifyConstant,
  simplifyCore,
  typed
});
export var symbolicEqual = createSymbolicEqual({
  OperatorNode,
  parse,
  simplify,
  typed
});
export var leafCount = createLeafCount({
  parse,
  typed
});
export var parser = createParser({
  Parser,
  typed
});
export var rationalize = createRationalize({
  bignumber,
  fraction,
  AccessorNode,
  ArrayNode,
  ConstantNode,
  FunctionNode,
  IndexNode,
  ObjectNode,
  OperatorNode,
  ParenthesisNode,
  SymbolNode,
  add,
  config,
  divide,
  equal,
  isZero,
  mathWithTransform,
  matrix,
  multiply,
  parse,
  pow,
  simplify,
  simplifyConstant,
  simplifyCore,
  subtract,
  typed
});
export var derivative = createDerivative({
  ConstantNode,
  FunctionNode,
  OperatorNode,
  ParenthesisNode,
  SymbolNode,
  config,
  equal,
  isZero,
  numeric,
  parse,
  simplify,
  typed
});
export var help = createHelp({
  Help,
  mathWithTransform,
  typed
});
_extends(math, {
  e,
  false: _false,
  fineStructure,
  i,
  Infinity: _Infinity,
  LN10,
  LOG10E,
  NaN: _NaN,
  null: _null,
  phi,
  SQRT1_2,
  sackurTetrode,
  tau,
  true: _true,
  'E': e,
  version,
  efimovFactor,
  LN2,
  pi,
  replacer,
  reviver,
  SQRT2,
  typed,
  'PI': pi,
  weakMixingAngle,
  abs,
  acos,
  acot,
  acsc,
  addScalar,
  arg,
  asech,
  asinh,
  atan,
  atanh,
  bigint,
  bitNot,
  boolean,
  clone,
  combinations,
  complex,
  conj,
  cos,
  cot,
  csc,
  cube,
  equalScalar,
  erf,
  exp,
  expm1,
  filter,
  flatten,
  forEach,
  format,
  getMatrixDataType,
  hex,
  im,
  isInteger,
  isNegative,
  isPositive,
  isZero,
  LOG2E,
  lgamma,
  log10,
  log2,
  map,
  multiplyScalar,
  not,
  number,
  oct,
  pickRandom,
  print,
  random,
  re,
  sec,
  sign,
  sin,
  size,
  splitUnit,
  square,
  string,
  subtractScalar,
  tan,
  toBest,
  typeOf,
  acosh,
  acsch,
  asec,
  bignumber,
  chain,
  combinationsWithRep,
  cosh,
  csch,
  dot,
  isNaN,
  isPrime,
  mapSlices,
  matrix,
  matrixFromFunction,
  multiply,
  ones,
  randomInt,
  reshape,
  sech,
  sinh,
  sparse,
  sqrt,
  squeeze,
  tanh,
  transpose,
  xgcd,
  zeros,
  acoth,
  asin,
  bin,
  concat,
  coth,
  ctranspose,
  diag,
  dotMultiply,
  equal,
  fraction,
  identity,
  isNumeric,
  kron,
  largerEq,
  leftShift,
  matrixFromRows,
  mode,
  nthRoot,
  numeric,
  prod,
  resize,
  rightArithShift,
  round,
  smaller,
  to,
  unaryMinus,
  unequal,
  xor,
  add,
  atan2,
  bitAnd,
  bitOr,
  bitXor,
  cbrt,
  compare,
  compareText,
  count,
  deepEqual,
  divideScalar,
  equalText,
  floor,
  gcd,
  hasNumericValue,
  hypot,
  larger,
  log,
  lsolve,
  matrixFromColumns,
  min,
  mod,
  nthRoots,
  nullish,
  or,
  partitionSelect,
  qr,
  rightLogShift,
  smallerEq,
  subset,
  subtract,
  trace,
  usolve,
  zpk2tf,
  catalan,
  compareNatural,
  composition,
  cross,
  det,
  diff,
  distance,
  dotDivide,
  index,
  intersect,
  invmod,
  lcm,
  log1p,
  lsolveAll,
  max,
  range,
  row,
  setCartesian,
  setDistinct,
  setIsSubset,
  setPowerset,
  slu,
  sort,
  unaryPlus,
  usolveAll,
  and,
  ceil,
  column,
  cumsum,
  fix,
  setDifference,
  setMultiplicity,
  setSymDifference,
  sum,
  inv,
  lup,
  pinv,
  pow,
  setIntersect,
  setUnion,
  sqrtm,
  vacuumImpedance,
  wienDisplacement,
  atomicMass,
  bohrMagneton,
  boltzmann,
  conductanceQuantum,
  coulomb,
  createUnit,
  deuteronMass,
  dotPow,
  electricConstant,
  elementaryCharge,
  expm,
  faraday,
  fft,
  gamma,
  gravitationConstant,
  hartreeEnergy,
  ifft,
  klitzing,
  loschmidt,
  magneticConstant,
  molarMass,
  molarPlanckConstant,
  neutronMass,
  nuclearMagneton,
  planckCharge,
  planckLength,
  planckTemperature,
  protonMass,
  quantumOfCirculation,
  reducedPlanckConstant,
  rydberg,
  secondRadiation,
  speedOfLight,
  stefanBoltzmann,
  thomsonCrossSection,
  avogadro,
  bohrRadius,
  coulombConstant,
  divide,
  electronMass,
  factorial,
  firstRadiation,
  gravity,
  inverseConductanceQuantum,
  lusolve,
  magneticFluxQuantum,
  molarMassC12,
  multinomial,
  parse,
  permutations,
  planckMass,
  polynomialRoot,
  resolve,
  setSize,
  simplifyConstant,
  solveODE,
  stirlingS2,
  unit,
  bellNumbers,
  compile,
  eigs,
  fermiCoupling,
  gasConstant,
  kldivergence,
  mean,
  molarVolume,
  planckConstant,
  quantileSeq,
  simplifyCore,
  variance,
  classicalElectronRadius,
  evaluate,
  median,
  simplify,
  symbolicEqual,
  corr,
  freqz,
  leafCount,
  mad,
  parser,
  rationalize,
  std,
  zeta,
  derivative,
  norm,
  rotationMatrix,
  help,
  planckTime,
  schur,
  rotate,
  sylvester,
  lyap,
  config
});
_extends(mathWithTransform, math, {
  mapSlices: createMapSlicesTransform({
    isInteger,
    typed
  }),
  filter: createFilterTransform({
    typed
  }),
  forEach: createForEachTransform({
    typed
  }),
  map: createMapTransform({
    typed
  }),
  or: createOrTransform({
    DenseMatrix,
    concat,
    equalScalar,
    matrix,
    typed
  }),
  and: createAndTransform({
    add,
    concat,
    equalScalar,
    matrix,
    not,
    typed,
    zeros
  }),
  concat: createConcatTransform({
    isInteger,
    matrix,
    typed
  }),
  max: createMaxTransform({
    config,
    isNaN,
    larger,
    numeric,
    typed
  }),
  print: createPrintTransform({
    add,
    matrix,
    typed,
    zeros
  }),
  sum: createSumTransform({
    add,
    config,
    numeric,
    typed
  }),
  bitAnd: createBitAndTransform({
    add,
    concat,
    equalScalar,
    matrix,
    not,
    typed,
    zeros
  }),
  min: createMinTransform({
    config,
    isNaN,
    numeric,
    smaller,
    typed
  }),
  nullish: createNullishTransform({
    deepEqual,
    flatten,
    matrix,
    size,
    typed
  }),
  subset: createSubsetTransform({
    add,
    matrix,
    typed,
    zeros
  }),
  bitOr: createBitOrTransform({
    DenseMatrix,
    concat,
    equalScalar,
    matrix,
    typed
  }),
  diff: createDiffTransform({
    bignumber,
    matrix,
    number,
    subtract,
    typed
  }),
  index: createIndexTransform({
    Index,
    getMatrixDataType
  }),
  range: createRangeTransform({
    bignumber,
    matrix,
    add,
    config,
    equal,
    isPositive,
    isZero,
    larger,
    largerEq,
    smaller,
    smallerEq,
    typed
  }),
  row: createRowTransform({
    Index,
    matrix,
    range,
    typed
  }),
  column: createColumnTransform({
    Index,
    matrix,
    range,
    typed
  }),
  cumsum: createCumSumTransform({
    add,
    typed,
    unaryPlus
  }),
  mean: createMeanTransform({
    add,
    divide,
    typed
  }),
  quantileSeq: createQuantileSeqTransform({
    add,
    bignumber,
    compare,
    divide,
    isInteger,
    larger,
    mapSlices,
    multiply,
    partitionSelect,
    smaller,
    smallerEq,
    subtract,
    typed
  }),
  variance: createVarianceTransform({
    add,
    divide,
    isNaN,
    mapSlices,
    multiply,
    subtract,
    typed
  }),
  std: createStdTransform({
    map,
    sqrt,
    typed,
    variance
  })
});
_extends(classes, {
  BigNumber,
  Complex,
  Fraction,
  Matrix,
  Node,
  ObjectNode,
  OperatorNode,
  ParenthesisNode,
  Range,
  RelationalNode,
  ResultSet,
  ArrayNode,
  BlockNode,
  ConditionalNode,
  ConstantNode,
  DenseMatrix,
  RangeNode,
  Chain,
  FunctionAssignmentNode,
  SparseMatrix,
  IndexNode,
  ImmutableDenseMatrix,
  Index,
  AccessorNode,
  AssignmentNode,
  FibonacciHeap,
  Spa,
  Unit,
  SymbolNode,
  FunctionNode,
  Help,
  Parser
});
Chain.createProxy(math);
export { embeddedDocs as docs } from '../expression/embeddedDocs/embeddedDocs.js';