#include "pywrapper.h"

#include <atomic>
#include <symengine/serialize-cereal.h>

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Divide PyNumber_TrueDivide
#endif

namespace SymEngine {

namespace {

std::atomic<bool> python_runtime_dead{false};

//! Ownership prefix reserved for Python callback-backed numeric domains.
constexpr const char *kPyNumberSubtypePrefix
    = "org.symengine.python.PyNumber/";

//! Semantic domain owned by the Python callback-backed wrapper.
constexpr const char *kPyFunctionSubtypeKey
    = "org.symengine.python.PyFunction";

const PyNumber &checked_py_number(const NumberWrapper &other)
{
#if HAVE_SYMENGINE_RTTI
    if (!is_a_number_wrapper<PyNumber>(other)) {
        throw SymEngineException(
            "PyNumber stable subtype key used by another C++ implementation");
    }
#endif
    return static_cast<const PyNumber &>(other);
}

const PyNumber *same_py_number(const PyNumber &self, const Number &other)
{
    if (!is_a<NumberWrapper>(other))
        return nullptr;
    const auto &wrapper = down_cast<const NumberWrapper &>(other);
    if (number_wrapper_type_compare(self, wrapper) != 0)
        return nullptr;
    return &checked_py_number(wrapper);
}

int checked_callable_compare(PyObject *lhs, PyObject *rhs, int operation,
                             const char *relation)
{
    const int result = PyObject_RichCompareBool(lhs, rhs, operation);
    if (result < 0) {
        // Do not let a Python exception survive beside the C++ refusal.  The
        // public wrapper boundary translates this SymEngineException into the
        // documented RuntimeError.
        PyErr_Clear();
        throw SymEngineException(std::string{"PyFunction callable "}
                                 + relation
                                 + " comparison raised an exception");
    }
    return result;
}

[[noreturn]] void python_callback_failed(const std::string &subject,
                                         const char *operation)
{
    PyErr_Clear();
    throw SymEngineException(subject + " " + operation
                             + " callback raised an exception");
}

int checked_python_compare(PyObject *lhs, PyObject *rhs, int operation,
                           const std::string &subject, const char *relation)
{
    const int result = PyObject_RichCompareBool(lhs, rhs, operation);
    if (result < 0)
        python_callback_failed(subject, relation);
    return result;
}

bool checked_symmetric_equality(PyObject *lhs, PyObject *rhs,
                                const std::string &subject)
{
    const int forward
        = checked_python_compare(lhs, rhs, Py_EQ, subject, "equality");
    const int reverse
        = checked_python_compare(rhs, lhs, Py_EQ, subject, "equality");
    if (forward != reverse)
        throw SymEngineException(subject + " equality is not symmetric");
    return forward == 1;
}

int checked_strict_order(PyObject *lhs, PyObject *rhs,
                         const std::string &subject)
{
    const int forward
        = checked_python_compare(lhs, rhs, Py_LT, subject, "less-than");
    const int reverse
        = checked_python_compare(rhs, lhs, Py_LT, subject, "less-than");
    if (forward == reverse) {
        throw SymEngineException(subject
                                 + (forward == 1
                                        ? " less-than order is inconsistent"
                                        : " unequal values are unordered"));
    }
    return forward == 1 ? -1 : 1;
}

hash_t checked_python_hash(PyObject *object, const char *subject)
{
    const Py_hash_t result = PyObject_Hash(object);
    if (result == -1) {
        PyErr_Clear();
        throw SymEngineException(std::string{subject}
                                 + " hash raised an exception");
    }
    return static_cast<hash_t>(result);
}

PyObject *checked_new_reference(PyObject *result, const std::string &subject,
                                const char *operation)
{
    if (result == nullptr || PyErr_Occurred() != nullptr) {
        Py_XDECREF(result);
        python_callback_failed(subject, operation);
    }
    return result;
}

PyObject *checked_to_python(const RCP<const PyModule> &module,
                            const RCP<const Basic> &value,
                            const char *operation)
{
    if (module.is_null())
        throw SymEngineException("PyNumber has no conversion module");
    return checked_new_reference(module->to_py_(value), "PyNumber",
                                 operation);
}

RCP<const Number>
checked_number_result(PyObject *result, const RCP<const PyModule> &module,
                      const char *operation)
{
    result = checked_new_reference(result, "PyNumber", operation);
    return make_rcp<PyNumber>(result, module);
}

PyObject *checked_initial_py_number(PyObject *object)
{
    if (object == nullptr)
        python_callback_failed("PyNumber", "construction");
    return object;
}

std::string
checked_py_number_subtype_key(const RCP<const PyModule> &pymodule)
{
    if (pymodule.is_null())
        throw SymEngineException("PyNumber requires a conversion module");
    const std::string &key = pymodule->get_number_subtype_key();
    const std::size_t prefix_size
        = std::char_traits<char>::length(kPyNumberSubtypePrefix);
    if (key.size() <= prefix_size
        || key.compare(0, prefix_size, kPyNumberSubtypePrefix) != 0) {
        throw SymEngineException(
            "PyModule PyNumber subtype key is not ownership-qualified");
    }
    return key;
}

void python_runtime_shutdown()
{
    python_runtime_dead.store(true, std::memory_order_relaxed);
}

void python_cooperative_incref(void *object) noexcept
{
    if (python_runtime_dead.load(std::memory_order_relaxed))
        return;
    PyGILState_STATE state = PyGILState_Ensure();
    Py_INCREF(reinterpret_cast<PyObject *>(object));
    PyGILState_Release(state);
}

void python_cooperative_decref(void *object) noexcept
{
    if (python_runtime_dead.load(std::memory_order_relaxed))
        return;
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(reinterpret_cast<PyObject *>(object));
    PyGILState_Release(state);
}

} // namespace

void initialize_python_cooperative_intrusive()
{
    static bool initialized = []() {
        cooperative_intrusive_init(python_cooperative_incref,
                                   python_cooperative_decref);
        Py_AtExit(python_runtime_shutdown);
        return true;
    }();
    (void)initialized;
}

bool is_a_PyNumber(const Basic &value) noexcept
{
    if (!is_a<NumberWrapper>(value))
        return false;
#if HAVE_SYMENGINE_RTTI
    return is_a_number_wrapper<PyNumber>(value);
#else
    const auto &wrapper = down_cast<const NumberWrapper &>(value);
    const std::string &key = wrapper.get_subtype_key();
    const std::size_t prefix_size
        = std::char_traits<char>::length(kPyNumberSubtypePrefix);
    return key.size() > prefix_size
           && key.compare(0, prefix_size, kPyNumberSubtypePrefix) == 0;
#endif
}

// PyModule
PyModule::PyModule(std::string number_subtype_key,
                   PyObject* (*to_py)(const RCP<const Basic>), RCP<const Basic> (*from_py)(PyObject*),
                   RCP<const Number> (*eval)(PyObject*, long), RCP<const Basic> (*diff)(PyObject*, RCP<const Basic>)) :
        number_subtype_key_(std::move(number_subtype_key)), to_py_(to_py),
        from_py_(from_py), eval_(eval), diff_(diff) {
    zero = PyInt_FromLong(0);
    one = PyInt_FromLong(1);
    minus_one = PyInt_FromLong(-1);
}

PyModule::~PyModule(){
    Py_DECREF(zero);
    Py_DECREF(one);
    Py_DECREF(minus_one);
}

// PyNumber
PyNumber::PyNumber(PyObject* pyobject, const RCP<const PyModule> &pymodule) :
        NumberWrapper(checked_py_number_subtype_key(pymodule)),
        pyobject_(checked_initial_py_number(pyobject)),
        pymodule_(pymodule) {
}

hash_t PyNumber::__hash__() const {
    return checked_python_hash(pyobject_, "PyNumber");
}

bool PyNumber::value_eq(const NumberWrapper &other) const {
    return checked_symmetric_equality(
        pyobject_, checked_py_number(other).get_py_object(), "PyNumber");
}

int PyNumber::value_compare(const NumberWrapper &other) const {
    PyObject* o1 = checked_py_number(other).get_py_object();
    return checked_strict_order(pyobject_, o1, "PyNumber");
}

bool PyNumber::is_zero() const {
    return checked_symmetric_equality(pyobject_, pymodule_->get_zero(),
                                      "PyNumber zero predicate");
}
//! \return true if `1`
bool PyNumber::is_one() const {
    return checked_symmetric_equality(pyobject_, pymodule_->get_one(),
                                      "PyNumber one predicate");
}
//! \return true if `-1`
bool PyNumber::is_minus_one() const {
    return checked_symmetric_equality(pyobject_, pymodule_->get_minus_one(),
                                      "PyNumber minus-one predicate");
}
//! \return true if negative
bool PyNumber::is_negative() const {
    if (is_zero())
        return false;
    return checked_strict_order(pyobject_, pymodule_->get_zero(),
                                "PyNumber sign predicate") < 0;
}
//! \return true if positive
bool PyNumber::is_positive() const {
    if (is_zero())
        return false;
    return checked_strict_order(pyobject_, pymodule_->get_zero(),
                                "PyNumber sign predicate") > 0;
}
//! \return true if complex
bool PyNumber::is_complex() const {
    return false;
}

//! Addition
RCP<const Number> PyNumber::add(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Add(pyobject_, other_p);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Add(pyobject_, other_p);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "addition");
}
//! Subtraction
RCP<const Number> PyNumber::sub(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Subtract(pyobject_, other_p);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Subtract(pyobject_, other_p);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "subtraction");
}
RCP<const Number> PyNumber::rsub(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Subtract(other_p, pyobject_);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Subtract(other_p, pyobject_);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "reverse subtraction");
}
//! Multiplication
RCP<const Number> PyNumber::mul(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Multiply(pyobject_, other_p);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Multiply(pyobject_, other_p);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "multiplication");
}
//! Division
RCP<const Number> PyNumber::div(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Divide(pyobject_, other_p);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Divide(pyobject_, other_p);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "division");
}
RCP<const Number> PyNumber::rdiv(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Divide(other_p, pyobject_);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Divide(other_p, pyobject_);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "reverse division");
}
//! Power
RCP<const Number> PyNumber::pow(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Power(pyobject_, other_p, Py_None);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Power(pyobject_, other_p, Py_None);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "power");
}
RCP<const Number> PyNumber::rpow(const Number &other) const {
    PyObject *other_p, *result;
    if (const PyNumber *py_number = same_py_number(*this, other)) {
        other_p = py_number->pyobject_;
        result = PyNumber_Power(other_p, pyobject_, Py_None);
    } else {
        other_p = checked_to_python(
            pymodule_, other.rcp_from_this_cast<const Basic>(), "conversion");
        result = PyNumber_Power(other_p, pyobject_, Py_None);
        Py_DECREF(other_p);
    }
    return checked_number_result(result, pymodule_, "reverse power");
}

RCP<const Number> PyNumber::eval(long bits) const {
    const RCP<const Number> result = pymodule_->eval_(pyobject_, bits);
    if (result.is_null() || PyErr_Occurred() != nullptr)
        python_callback_failed("PyNumber", "numeric conversion");
    return result;
}

std::string PyNumber::__str__() const {
    Py_ssize_t size;
    PyObject *pystr
        = checked_new_reference(PyObject_Str(pyobject_), "PyNumber", "string");
    const char* data = PyUnicode_AsUTF8AndSize(pystr, &size);
    if (data == nullptr || PyErr_Occurred() != nullptr) {
        Py_DECREF(pystr);
        python_callback_failed("PyNumber", "UTF-8 string conversion");
    }
    std::string result = std::string(data, size);
    Py_DECREF(pystr);
    return result;
}

// PyFunctionClass

PyFunctionClass::PyFunctionClass(PyObject *pyobject, std::string name, const RCP<const PyModule> &pymodule) :
        pyobject_{pyobject}, name_{name}, pymodule_{pymodule} {

}

PyObject* PyFunctionClass::call(const vec_basic &vec) const {
    PyObject *tuple = PyTuple_New(vec.size());
    for (unsigned i = 0; i < vec.size(); i++) {
        PyTuple_SetItem(tuple, i, pymodule_->to_py_(vec[i]));
    }
    PyObject* result = PyObject_CallObject(pyobject_, tuple);
    Py_DECREF(tuple);
    return result;
}

bool PyFunctionClass::__eq__(const PyFunctionClass &x) const {
    const int forward
        = checked_callable_compare(pyobject_, x.pyobject_, Py_EQ, "equality");
    const int reverse
        = checked_callable_compare(x.pyobject_, pyobject_, Py_EQ, "equality");
    if (forward != reverse)
        throw SymEngineException(
            "PyFunction callable equality is not symmetric");
    return forward == 1;
}

int PyFunctionClass::compare(const PyFunctionClass &x) const {
    if (__eq__(x)) return 0;
    const int forward = checked_callable_compare(pyobject_, x.pyobject_, Py_LT,
                                                 "strict-order");
    const int reverse = checked_callable_compare(x.pyobject_, pyobject_, Py_LT,
                                                 "strict-order");
    if (forward == reverse) {
        if (forward == 1)
            throw SymEngineException(
                "PyFunction callable strict order is not antisymmetric");
        throw SymEngineException(
            "Unequal PyFunction callables do not define a strict total order");
    }
    return forward == 1 ? -1 : 1;
}

// PyFunction
PyFunction::PyFunction(const vec_basic &vec, const RCP<const PyFunctionClass> &pyfunc_class,
           PyObject *pyobject) : FunctionWrapper(pyfunc_class->get_name(), std::move(vec),
                                                 kPyFunctionSubtypeKey),
           pyfunction_class_{pyfunc_class}, pyobject_{pyobject} {

}

PyFunction::~PyFunction() {
    Py_DECREF(pyobject_);
}

PyObject* PyFunction::get_py_object() const {
    return pyobject_;
}

RCP<const PyFunctionClass> PyFunction::get_pyfunction_class() const {
    return pyfunction_class_;
}

RCP<const Basic> PyFunction::create(const vec_basic &x) const {
    PyObject* pyobj = pyfunction_class_->call(x);
    RCP<const Basic> result = pyfunction_class_->get_py_module()->from_py_(pyobj);
    Py_XDECREF(pyobj);
    return result;
}

RCP<const Number> PyFunction::eval(long bits) const {
    return pyfunction_class_->get_py_module()->eval_(pyobject_, bits);
}

RCP<const Basic> PyFunction::diff_impl(const RCP<const Symbol> &s) const {
    return pyfunction_class_->get_py_module()->diff_(pyobject_, s);
}

hash_t PyFunction::__hash__() const {
    // Python's application hash follows the callable-and-arguments equality
    // refined below; the stable subtype key is constant throughout this domain.
    return checked_python_hash(pyobject_, "PyFunction application");
}

bool PyFunction::__eq__(const Basic &o) const {
    if (!is_a<FunctionWrapper>(o)) return false;
    const FunctionWrapper &wrapper = down_cast<const FunctionWrapper &>(o);
    if (get_name() != wrapper.get_name()
        or function_wrapper_type_compare(*this, wrapper) != 0)
        return false;
#if HAVE_SYMENGINE_RTTI
    if (!is_a_function_wrapper<PyFunction>(o)) return false;
#endif
    const PyFunction &other = static_cast<const PyFunction &>(o);
    return pyfunction_class_->__eq__(*other.get_pyfunction_class())
           and unified_eq(get_vec(), other.get_vec());
}

int PyFunction::compare(const Basic &o) const {
    SYMENGINE_ASSERT(is_a<FunctionWrapper>(o))
    const FunctionWrapper &wrapper = down_cast<const FunctionWrapper &>(o);
    if (get_name() != wrapper.get_name())
        return get_name() < wrapper.get_name() ? -1 : 1;
    const int type_cmp = function_wrapper_type_compare(*this, wrapper);
    if (type_cmp != 0) return type_cmp;
#if HAVE_SYMENGINE_RTTI
    if (!is_a_function_wrapper<PyFunction>(o))
        throw SymEngineException(
            "PyFunction stable subtype key used by another C++ implementation");
#endif
    const PyFunction &s = static_cast<const PyFunction &>(o);
    int cmp = pyfunction_class_->compare(*s.get_pyfunction_class());
    if (cmp != 0) return cmp;
    return unified_compare(get_vec(), s.get_vec());
}

inline PyObject* get_pickle_module() {
    static PyObject *module = NULL;
    if (module == NULL) {
        module = PyImport_ImportModule("pickle");
    }
    if (module == NULL) {
        throw SymEngineException("error importing pickle module.");
    }
    return module;
}

PyObject* pickle_loads(const std::string &pickle_str) {
    PyObject *module = get_pickle_module();
    PyObject *pickle_bytes = PyBytes_FromStringAndSize(pickle_str.data(), pickle_str.size());
    PyObject *obj = PyObject_CallMethod(module, "loads", "O", pickle_bytes);
    Py_XDECREF(pickle_bytes);
    if (obj == NULL) {
        throw SerializationError("error when loading pickled symbol subclass object");
    }
    return obj;
}

RCP<const Basic> load_basic(RCPBasicAwareInputArchive<cereal::PortableBinaryInputArchive> &ar, RCP<const Symbol> &)
{
    bool is_pysymbol;
    bool store_pickle;
    std::string name;
    ar(is_pysymbol);
    ar(name);
    if (is_pysymbol) {
        std::string pickle_str;
        ar(pickle_str);
        ar(store_pickle);
        PyObject *obj = pickle_loads(pickle_str);
        // The deserialization-local Python object cannot serve as a durable
        // weak-reference target.  Preserve its pickle bytes in the temporary
        // native marker; c2py() will reconstruct and externalize the wrapper.
        RCP<const Basic> result = make_rcp<PySymbol>(name, obj, true);
        Py_XDECREF(obj);
        return result;
    } else {
        return symbol(name);
    }
}

std::string pickle_dumps(const PyObject * obj) {
    PyObject *module = get_pickle_module();
    PyObject *pickle_bytes = PyObject_CallMethod(module, "dumps", "O", obj);
    if (pickle_bytes == NULL) {
        throw SerializationError("error when pickling symbol subclass object");
    }
    Py_ssize_t size;
    char* buffer;
    PyBytes_AsStringAndSize(pickle_bytes, &buffer, &size);
    return std::string(buffer, size);
}

void save_basic(RCPBasicAwareOutputArchive<cereal::PortableBinaryOutputArchive> &ar, const Symbol &b)
{
    bool is_pysymbol = is_a_sub<PySymbol>(b);
    ar(is_pysymbol);
    ar(b.__str__());
    if (is_pysymbol) {
        RCP<const PySymbol> p = rcp_static_cast<const PySymbol>(b.rcp_from_this());
        PyObject *obj = p->get_py_object();
        std::string pickle_str = pickle_dumps(obj);
        ar(pickle_str);
        ar(p->store_pickle);
        Py_XDECREF(obj);
    }
}

std::string wrapper_dumps(const Basic &x)
{
    std::ostringstream oss;
    unsigned short major = SYMENGINE_MAJOR_VERSION;
    unsigned short minor = SYMENGINE_MINOR_VERSION;
    RCPBasicAwareOutputArchive<cereal::PortableBinaryOutputArchive>{oss}(major, minor,
                                             x.rcp_from_this());
    return oss.str();
}

RCP<const Basic> wrapper_loads(const std::string &serialized)
{
    unsigned short major, minor;
    RCP<const Basic> obj;
    std::istringstream iss(serialized);
    RCPBasicAwareInputArchive<cereal::PortableBinaryInputArchive> iarchive{iss};
    iarchive(major, minor);
    if (major != SYMENGINE_MAJOR_VERSION or minor != SYMENGINE_MINOR_VERSION) {
        throw SerializationError(StreamFmt()
                                 << "SymEngine-" << SYMENGINE_MAJOR_VERSION
                                 << "." << SYMENGINE_MINOR_VERSION
                                 << " was asked to deserialize an object "
                                 << "created using SymEngine-" << major << "."
                                 << minor << ".");
    }
    iarchive(obj);
    return obj;
}

} // SymEngine
