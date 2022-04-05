/**
 * @file vf_pymodule.c
 * @author hangzhongling (hangzhongling@google.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-19
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "libavutil/common.h"
#include "libavutil/opt.h"
#include "libavutil/base64.h"
#include "libavutil/imgutils.h"
#include "libavutil/parseutils.h"
#include "libavutil/avstring.h"
#include "libavformat/avio.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <structmember.h>

#define av_strncpy(dst, src, size) \
 do { *dst = '\0' ; \
    if(size) { strncat(dst, src, size); } \
 } while(0)

#define TIMEBASE_MS (AVRational){1, 1000000}
#define MAX_FRAME_OUT 256

#define MODE_DEFAULT      0
#define MODE_ONE_TO_ONE   1 // support super resolution
#define MODE_ONE_TO_MANY  2 // support frc

typedef struct PyFrameObject {
    PyObject_HEAD
    PyObject *arrays;
    AVFrame *frame_data;
    int width;
    int height;
    int64_t pts;
    int64_t dts;
    const char *pixfmt_desc;
    const char *range_desc;
} PyFrameObject;

static void PyFrameObject_dealloc(PyFrameObject *self) 
{
    if(self->frame_data) {
        av_frame_free(&self->frame_data);
        self->frame_data = NULL;
    }
    Py_DECREF(self->arrays);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *PyFrameObject_arrays(PyFrameObject *self) 
{
    Py_INCREF(self->arrays);
    return self->arrays;
}

static PyObject *PyFrameObject_set_prop(PyFrameObject *self, PyObject *args) 
{
    char *key, *val;
    AVDictionary *meta = NULL;

    if(!PyArg_ParseTuple(args, "ss", &key, &val)) {
        PyErr_SetString(PyExc_ValueError, "invalid arg from");
        return NULL;
    }

    if(av_dict_set(&meta, key, val, 0) < 0) {
        PyErr_SetString(PyExc_ValueError, "av_dict error");
        return NULL;
    }
    av_frame_set_metadata(self->frame_data, meta);
    Py_INCREF(Py_None);

    return Py_None;
}

static PyObject *PyFrameObject_get_prop(PyFrameObject *self, PyObject *args) 
{
    char *key;
    AVDictionary *meta;
    AVDictionaryEntry *val;

    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_ValueError, "invalid arg form");
        return NULL;
    }

    meta = av_frame_get_metadata(self->frame_data);
    val = av_dict_get(meta, key, NULL, 0);
    if(!val) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyUnicode_FromString(val->value);
}

static PyObject *PyFrameObject_clone(PyFrameObject *self);

static PyMethodDef PyFrameObject_methods[] = {
    {"arrays", (PyCFunction)PyFrameObject_arrays, METH_NOARGS, "return an ndarray list"    },
    {"set_prop", (PyCFunction)PyFrameObject_set_prop, METH_VARARGS, "set frame property"   },
    {"get_prop", (PyCFunction)PyFrameObject_get_prop, METH_VARARGS, "get frame property"   },
    {"clone", (PyCFunction)PyFrameObject_clone, METH_NOARGS, "clone frame"                 },
    {NULL}
};

static PyMemberDef PyFrameObject_members[] = {
    {"width",       T_INT,         offsetof(PyFrameObject, width),   0,   "frame width"      },
    {"height",      T_INT,         offsetof(PyFrameObject, height),  0,   "frame height"    },
    {"pts",         T_LONGLONG,    offsetof(PyFrameObject, pts),     0,   "frame pts"    },
    {"dts",         T_LONGLONG,    offsetof(PyFrameObject, dts),     0,   "frame dts"    },
    {"ndarrays",    T_OBJECT,      offsetof(PyFrameObject, arrays),  0,   "frame data"    },
    {"pixfmt",      T_STRING,      offsetof(PyFrameObject, dts),     0,   "frame pixfmt"    },
    {"color_range", T_STRING,      offsetof(PyFrameObject, dts),     0,   "frame color range"    },
    { NULL }
};

static PyTypeObject pymodule_PyFrameObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      =  "pymodule.Frame",
    .tp_doc       =  "Frame in pymodule",
    .tp_basicsize = sizeof(PyFrameObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor) PyFrameObject_dealloc,
    .tp_methods   = PyFrameObject_methods,
    .tp_members   = PyFrameObject_members,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = PyType_GenericNew,
};

static PyObject *PyFrameObject_alloc(AVFilterLink *link, AVFrame *in) 
{
    //TODO: move some of this to ctx
    const AVPixFmtDescriptor *desc;
    int max_step[4];
    int max_step_comp[4];
    PyFrameObject *frame;
    PyObject *arrays = PyList_New(0);

    desc = av_pix_fmt_desc_get(in->format);
    av_image_fill_max_pixsteps(max_step, max_step_comp, desc);
    for(int i = 0; i < 4; i++) {
        static const struct {
            int type, size;
        } dtype_map[3] = {
            {NPY_UINT8, 1},
            {NPY_UINT16, 2},
            {NPY_UINT32, 4}
        };
        if(max_step[i]) {
            PyObject* array;
            int w = in->linesize[i];
            int h = in->height;
            int s = (desc->comp[i].depth - 1) >> 3;
            if(i == 1 || i == 2) {
                h = AV_CEIL_RSHIFT(h, desc->log2_chroma_h);
            }

            npy_intp dims[3] = {
                h, w / max_step[i], max_step[i] / dtype_map[s].size
            };
            array = PyArray_SimpleNewFromData(
                3, dims, dtype_map[s].type, in->data[i]);
            PyList_Append(arrays, array);
            Py_DECREF(array);
        }
    }

    frame = (PyFrameObject *) PyType_GenericNew(
        &pymodule_PyFrameObjectType, NULL, NULL);
    frame->arrays      = arrays;
    frame->frame_data  = in;
    frame->width       = in->width;
    frame->height      = in->height;
    frame->pts         = av_rescale_q(in->pts, link->time_base, TIMEBASE_MS);
    frame->dts         = av_rescale_q(in->pkt_dts, link->time_base, TIMEBASE_MS);
    frame->pixfmt_desc = av_get_pix_fmt_name(in->format);
    frame->range_desc  = av_color_range_name(in->color_range);

    return (PyObject *) frame;
}

static PyObject *PyFrameObject_clone(PyFrameObject *self) 
{
    int ret;
    const AVPixFmtDescriptor *desc;
    int max_step[4];
    int max_step_comp[4];
    AVFrame *cloned;
    PyFrameObject *frame;
    PyObject *arrays;

    AVFrame* in = self->frame_data; 
    cloned = av_frame_clone(in);
    if (!cloned) {
        PyErr_SetString(PyExc_MemoryError, "clone AVFrame failed");
        return NULL;
    }
    ret = av_frame_make_writable(cloned);
    if(ret < 0) {
        PyErr_SetString(PyExc_RuntimeError, "mutate AVFrame failed");
        return NULL;
    }

    arrays = PyList_New(0);
    desc = av_pix_fmt_desc_get(in->format);
    av_image_fill_max_pixsteps(max_step, max_step_comp, desc);
    for (int i = 0; i < 4; i++) {
        static const struct {
            int type, size;
        } dtype_map[3] = {
            {NPY_UINT8, 1},
            {NPY_UINT16, 2},
            {NPY_UINT32, 4}
        };

        if (max_step[i]) {
            PyObject* array;
            int w = cloned->width;
            int h = cloned->height;
            int s = (desc->comp[i].depth - 1) >> 3;
            if(i == 1 || 1 == 2) {
                h = AV_CEIL_RSHIFT(h, desc->log2_chroma_h);
            }
            npy_intp dims[3] = {
                h, w / max_step[i], max_step[i] / dtype_map[s].size
            };
            array = PyArray_SimpleNewFromData(
                3, dims, dtype_map[s].type, in->data[i]);
            PyList_Append(arrays, array);
            Py_DECREF(array);
        }
    }

    frame = (PyFrameObject *) PyType_GenericNew(
        &pymodule_PyFrameObjectType, NULL, NULL);
    frame->arrays      = arrays;
    frame->frame_data  = cloned;
    frame->width       = self->width;
    frame->height      = self->height;
    frame->pts         = self->pts;
    frame->dts         = self->dts;
    frame->pixfmt_desc = self->pixfmt_desc;
    frame->range_desc  = self->range_desc;

    return (PyObject *) frame;
}

typedef struct PyModuleContext {
    const AVClass *class;
    char *module_path;
    char *module_opts;
    PyObject *module;
    PyObject *setup_args;
    PyObject *process_frame_args;
    enum AVPixelFormat *formats;
    int nb_formats;
    int process_mode;
} PyModuleContext;

static int process_frame(AVFilterLink *inlink, AVFrame *in) 
{
    PyObject *func, *value;
    AVFilterContext *ctx = inlink->dst;
    PyModuleContext *s = ctx->priv;

    func = PyObject_GetAttrString(s->module, "process_frame");
    if (!func) {
        av_log(s, AV_LOG_ERROR, "process_frame() not found in %s\n", s->module_path);
        return AVERROR(EINVAL);
    }
    PyObject *frame = PyFrameObject_alloc(inlink, in);
    PyObject *argin = PyTuple_New(1);
    PyTuple_SetItem(argin, 0, frame);

    value = PyObject_Call(func, argin, s->process_frame_args);
    ((PyFrameObject *)frame)->frame_data = NULL; // move ownership back
    Py_DECREF(argin);
    Py_DECREF(func);

    if (!value || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }
    if(!PyLong_Check(value)) {
        av_log(s, AV_LOG_ERROR, "process_frame() should return int\n");
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }
    if(PyLong_AsLong(value) < 0) {
        av_log(s, AV_LOG_ERROR, "process_frame() return error %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }
    Py_DECREF(value);
    return 0;
}

static int process_frame_one_to_one(AVFilterLink *inlink,  AVFrame *in, 
                                    AVFilterLink *outlink, AVFrame *out, AVFrame *outs[MAX_FRAME_OUT]) 
{
    PyObject *func, *value;
    Py_ssize_t n_outs;
    AVFilterContext *ctx = inlink->dst;
    PyModuleContext *s = ctx->priv;

    func = PyObject_GetAttrString(s->module, "process_frame");
    if(!func) {
        av_log(s, AV_LOG_ERROR, "process_frame() not found in %s\n", s->module_path);
        return AVERROR(EINVAL);
    }

    PyObject *iframe = PyFrameObject_alloc(inlink, in);
    PyObject *oframe = PyFrameObject_alloc(outlink, out);
    PyObject *argin = PyTuple_New(2);
    PyTuple_SetItem(argin, 0, iframe);
    PyTuple_SetItem(argin, 1, oframe);

    value = PyObject_Call(func, argin, s->process_frame_args);
    Py_DECREF(argin);
    Py_DECREF(func);

    if (!value || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }

    if(!PyList_Check(value)) {
        av_log(s, AV_LOG_WARNING, "process_frame() should return a valid frame list %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }

    if((n_outs = PyList_Size(value)) > MAX_FRAME_OUT) {
        av_log(s, AV_LOG_WARNING, "process_frame() return too many frames %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }

    for(int k = 0; k < n_outs; k++) {
        int ret;
        AVFrame *frame_ref;
        PyFrameObject *object;

        object = (PyFrameObject *)PyList_GetItem(value, k);
        if(!PyObject_TypeCheck(object, &pymodule_PyFrameObjectType)) {
            av_log(s, AV_LOG_WARNING, "process_frame() not a valid frame list %s\n", s->module_path);
            Py_DECREF(value);
            return AVERROR(EINVAL);
        }

        frame_ref = av_frame_alloc();
        if(!frame_ref) {
            return AVERROR(EINVAL);
        }
        ret = av_frame_ref(frame_ref, object->frame_data);
        if(ret < 0) {
            return ret;
        }
        outs[k] = frame_ref;
        outs[k]->pts = av_rescale_q(object->pts, TIMEBASE_MS, outlink->time_base);
        outs[k]->pkt_dts = av_rescale_q(object->dts, TIMEBASE_MS, outlink->time_base);
    }
    Py_DECREF(value);
    return n_outs;
}

static int process_frame_one_to_many(AVFilterLink *inlink, AVFrame *in, AVFrame *outs[MAX_FRAME_OUT]) 
{
    PyObject *func, *value;
    Py_ssize_t n_outs;
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    PyModuleContext *s = ctx->priv;

    func = PyObject_GetAttrString(s->module, "process_frame");
    if(!func) {
        av_log(s, AV_LOG_ERROR, "process_frame() not found in %s\n", s->module_path);
        return AVERROR(EINVAL);
    }

    PyObject *iframe = PyFrameObject_alloc(inlink, in);
    PyObject *argin = PyTuple_New(1);
    PyTuple_SetItem(argin, 0, iframe);

    value = PyObject_Call(func, argin, s->process_frame_args);
    Py_DECREF(argin);
    Py_DECREF(func);

    if (!value || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }

    if(!PyList_Check(value)) {
        av_log(s, AV_LOG_WARNING, "process_frame() should return a valid frame list %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }

    if((n_outs = PyList_Size(value)) > MAX_FRAME_OUT) {
        av_log(s, AV_LOG_WARNING, "process_frame() return too many frames %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }
    
    for(int k = 0; k < n_outs; k++) {
        int ret;
        AVFrame *frame_ref;
        PyFrameObject *object;

        object = (PyFrameObject *)PyList_GetItem(value, k);
        if(!PyObject_TypeCheck(object, &pymodule_PyFrameObjectType)) {
            av_log(s, AV_LOG_WARNING, "process_frame() not a valid frame list %s\n", s->module_path);
            Py_DECREF(value);
            return AVERROR(EINVAL);
        }

        frame_ref = av_frame_alloc();
        if(!frame_ref) {
            return AVERROR(EINVAL);
        }
        ret = av_frame_ref(frame_ref, object->frame_data);
        if(ret < 0) {
            return ret;
        }
        outs[k] = frame_ref;
        outs[k]->pts = av_rescale_q(object->pts, TIMEBASE_MS, outlink->time_base);
        outs[k]->pkt_dts = av_rescale_q(object->dts, TIMEBASE_MS, outlink->time_base);
    }
    Py_DECREF(value);
    return n_outs;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) 
{
    int ret;
    AVFrame *processed;
    AVFilterContext *ctx = inlink->dst;
    PyModuleContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    ret = av_frame_make_writable(in);
    if (ret < 0) {
        return AVERROR(ENOMEM);
    }
    if (s->process_mode == MODE_ONE_TO_MANY) {
        int n_out;
        AVFrame *outs[MAX_FRAME_OUT] = {};
        n_out = process_frame_one_to_many(inlink, in, outs);
        if (n_out < 0) {
            return AVERROR(EINVAL);
        }
        for(int k = 0; k < n_out; k++) {
            ret = ff_filter_frame(inlink->dst->outputs[0], outs[k]);
            if (ret < 0) {
                return ret;
            }
        }
        return 0;
    } else if (s->process_mode == MODE_ONE_TO_ONE) {
        AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if(!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
        int n_out;
        AVFrame *outs[MAX_FRAME_OUT] = {};

        n_out = process_frame_one_to_one(inlink, in, outlink, out, outs);
        if(n_out < 0) {
            return AVERROR(EINVAL);
        }
        for(int k = 0; k < n_out; k++) {
            ret = ff_filter_frame(inlink->dst->outputs[0], outs[k]);
            if(ret < 0) {
                return ret;
            }
        }

        return 0;
    } else {
        ret = process_frame(inlink, in);
        if(ret < 0) {
            return ret;
        }
        processed = in;
    }
    return ff_filter_frame(inlink->dst->outputs[0], processed);
} 

static int config_input(AVFilterLink *inlink) 
{
    int ret;
    PyModuleContext *s = inlink->dst->priv;

    if(s->module_opts) {
        char *key, *value;
        const char *opts = s->module_opts;

        s->setup_args = PyDict_New();

        while(*opts) {
            ret = av_opt_get_key_value(&opts, "=", ",", 0, &key, &value);
            if(ret < 0) {
                return ret;
            }
            if(*opts)
                opts++;
            av_log(s, AV_LOG_DEBUG, "opts get key: %s, value: %s\n", key, value);

            if(!av_strncasecmp(value, "n", 1) && !strncmp(value+1, "one", 3)) {
                PyObject *k = Py_BuildValue("s", key);
                PyDict_SetItem(s->setup_args, k, Py_None);
                Py_XDECREF(k);
            } else if (!av_strncasecmp(value, "t", 1) && !strncmp(value+1, "rue", 3)) {
                PyObject *k = Py_BuildValue("s", key);
                PyObject *v = Py_BuildValue("O", Py_True);
                PyDict_SetItem(s->setup_args, k, v);
                Py_XDECREF(k);
                Py_XDECREF(v);
            }  else if (!av_strncasecmp(value, "f", 1) && !strncmp(value+1, "alse", 4)) {
                PyObject *k = Py_BuildValue("s", key);
                PyObject *v = Py_BuildValue("O", Py_False);
                PyDict_SetItem(s->setup_args, k, v);
                Py_XDECREF(k);
                Py_XDECREF(v);
            }
            av_free(value);
            av_free(key);
        }
    }
    return 0;
}

static int flush_frames(AVFilterLink *inlink, AVFrame *outs[MAX_FRAME_OUT]) 
{   
    PyObject *func, *value;
    Py_ssize_t n_outs;
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    PyModuleContext *s = ctx->priv;

    func = PyObject_GetAttrString(s->module, "flush_frames");
    if(!func) {
        av_log(s, AV_LOG_ERROR, "flush_frames() not found in %s\n", s->module_path);
        return AVERROR(EINVAL);
    }

    PyObject *argin = PyTuple_New(0);

    value = PyObject_Call(func, argin, s->process_frame_args);
    Py_DECREF(argin);
    Py_DECREF(func);

    if (!value || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }

    if(!PyList_Check(value)) {
        av_log(s, AV_LOG_WARNING, "flush_frames() should return a valid frame list %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }

    if((n_outs = PyList_Size(value)) > MAX_FRAME_OUT) {
        av_log(s, AV_LOG_WARNING, "flush_frames() return too many frames %s\n", s->module_path);
        Py_DECREF(value);
        return AVERROR(EINVAL);
    }

    for(int k = 0; k < n_outs; k++) {
        int ret;
        AVFrame *frame_ref;
        PyFrameObject *object;

        object = (PyFrameObject *)PyList_GetItem(value, k);
        if(!PyObject_TypeCheck(object, &pymodule_PyFrameObjectType)) {
            av_log(s, AV_LOG_WARNING, "flush_frames() not a valid frame list %s\n", s->module_path);
            Py_DECREF(value);
            return AVERROR(EINVAL);
        }

        frame_ref = av_frame_alloc();
        if(!frame_ref) {
            return AVERROR(EINVAL);
        }
        ret = av_frame_ref(frame_ref, object->frame_data);
        if(ret < 0) {
            return ret;
        }
        outs[k] = frame_ref;
        outs[k]->pts = av_rescale_q(object->pts, TIMEBASE_MS, outlink->time_base);
        outs[k]->pkt_dts = av_rescale_q(object->dts, TIMEBASE_MS, outlink->time_base);
    }
    Py_DECREF(value);
    return n_outs;
}

static int request_frame(AVFilterLink *outlink) 
{
    int ret;
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    PyModuleContext *s = ctx->priv;

    ret = ff_request_frame(inlink);
    if(ret == AVERROR_EOF) {
        if(s->process_mode == MODE_ONE_TO_MANY || s->process_mode == MODE_ONE_TO_ONE) {
            int n_out;
            AVFrame *outs[MAX_FRAME_OUT] = {};
            n_out = flush_frames(inlink, outs);
            if(n_out < 0) {
                return AVERROR(EINVAL);
            }
            for(int k = 0; k < n_out; k++) {
                int res = ff_filter_frame(outlink, outs[k]);
                if(res < 0) {
                    return res;
                }
            }
        }
    }
    return ret;
}

static int config_output(AVFilterLink *outlink) {
    const AVPixFmtDescriptor *desc;
    PyObject *func, *tuple, *dict;
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    PyModuleContext *s = ctx->priv;

    outlink->format = inlink->format;
    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->time_base = inlink->time_base;
    outlink->frame_rate = inlink->frame_rate;

    func = PyObject_GetAttrString(s->module, "setup");
    if(!func) {
        av_log(s, AV_LOG_WARNING, "setup() not found in %s\n", s->module_path);
        return 0;
    }

    tuple = Py_BuildValue("(i, i, s)",
        inlink->w, inlink->h, av_pix_fmt_desc_get(inlink->format)->name);
    dict = PyObject_Call(func, tuple, s->setup_args);
    Py_DECREF(func);
    Py_DECREF(tuple);

    if(!dict || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }

    if(PyDict_Check(dict)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while(PyDict_Next(dict, &pos, &key, &value)) {
            PyObject *bytes = PyUnicode_AsEncodedString(key, "UTF-8", "strict");
            if(!strcmp(PyBytes_AS_STRING(bytes), "config")) {
                if(PyDict_Check(value)) {
                    PyObject *key;
                    key = PyUnicode_FromString("w");
                    if(PyDict_Contains(value, key)) {
                        PyObject *data = PyDict_GetItemString(value, "w");
                        if(!PyLong_CheckExact(data)) {
                            av_log(s, AV_LOG_ERROR, "config contains invalid 'w' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        outlink->w = PyLong_AsLong(data);
                    }
                    Py_DECREF(key);
                    key = PyUnicode_FromString("h");
                    if(PyDict_Contains(value, key)) {
                        PyObject *data = PyDict_GetItemString(value, "h");
                        if(!PyLong_CheckExact(data)) {
                            av_log(s, AV_LOG_ERROR, "config contains invalid 'h' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        outlink->h = PyLong_AsLong(data);
                    }
                    Py_DECREF(key);
                    key = PyUnicode_FromString("pixfmt");
                    if(PyDict_Contains(value, key)) {
                        PyObject *data = PyDict_GetItemString(value, "pixfmt");
                        if(!PyUnicode_CheckExact(data)) {
                            av_log(s, AV_LOG_ERROR, "config contains invalid 'pixfmt' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        PyObject *bytes = PyUnicode_AsEncodedString(data, "UTF-8", "strict");
                        if(bytes) {
                            enum AVPixelFormat pixfmt = av_get_pix_fmt(PyBytes_AS_STRING(bytes));
                            if(pixfmt == AV_PIX_FMT_NONE) {
                                av_log(s, AV_LOG_ERROR, "config error in 'pixfmt' %s\n", s->module_path);
                                return AVERROR(EINVAL);
                            } else {
                                outlink->format = pixfmt;
                            }
                        }
                        Py_DECREF(bytes);
                    }
                    Py_DECREF(key);
                    key = PyUnicode_FromString("fr_ratio");
                    if(PyDict_Contains(value, key)) {
                        AVRational ratio;
                        PyObject *data = PyDict_GetItemString(value, "fr_ratio");
                        if(!PyNumber_Check(data)) {
                            av_log(s, AV_LOG_ERROR, "config contains invalid 'fr_ratio' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        PyObject *pynum = PyNumber_Float(data);
                        char *decimal = av_asprintf("%lf", PyFloat_AS_DOUBLE(pynum));
                        Py_DECREF(pynum);

                        if(av_parse_ratio_quiet(&ratio, decimal, 1001000) < 0 ||
                            ratio.num <= 0 ||
                            ratio.den <= 0) {
                            av_log(s, AV_LOG_ERROR, "config error with 'fr_ratio' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        av_free(decimal);
                        outlink->time_base = av_div_q(outlink->time_base, ratio);
                        outlink->frame_rate = av_mul_q(outlink->frame_rate, ratio);
                    }
                    Py_DECREF(key);
                    key = PyUnicode_FromString("process_mode");
                    if(PyDict_Contains(value, key)) {
                        PyObject *data = PyDict_GetItemString(value, "process_mode");
                        if(!PyUnicode_CheckExact(data)) {
                            av_log(s, AV_LOG_ERROR, "config contains invalid 'process_mode' %s\n", s->module_path);
                            return AVERROR(EINVAL);
                        }
                        PyObject *bytes = PyUnicode_AsEncodedString(data, "UTF-8", "strict");
                        if(bytes) {
                            const char *mode = PyBytes_AS_STRING(bytes);
                            if(!strcmp(mode, "one_to_one")) {
                                s->process_mode = MODE_ONE_TO_ONE;
                            } else if (!strcmp(mode, "one_to_many")) {
                                s->process_mode = MODE_ONE_TO_MANY;
                            }
                        }
                        Py_DECREF(bytes);
                    }
                    Py_DECREF(key);
                }
            } else {
                if(!s->process_frame_args) {
                    s->process_frame_args = PyDict_New();
                }
                PyDict_SetItem(s->process_frame_args, key, value);
            }
            Py_DECREF(bytes);
        }
    }
    Py_DECREF(dict);
    desc = av_pix_fmt_desc_get(outlink->format);
    if((desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) ||
        (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
        av_log(s, AV_LOG_ERROR, "don't support %s\n", desc->name);
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold int init(AVFilterContext *ctx) 
{
    PyObject *name;
    PyModuleContext *s = ctx->priv;

    Py_Initialize();
    import_array1(AVERROR(ENOSYS));

    name = PyUnicode_FromString(s->module_path);
    if(!name) {
        av_log(s, AV_LOG_ERROR, "python %s not found\n", s->module_path);
        return AVERROR(EINVAL);
    }

    s->module = PyImport_Import(name);
    Py_DECREF(name);
    if(!s->module || PyErr_Occurred()) {
        PyErr_Print();
        av_log(s, AV_LOG_ERROR, "load python %s failed\n", s->module_path);
        return AVERROR(EINVAL);
    }

    if(PyType_Ready(&pymodule_PyFrameObjectType) < 0) {
        av_log(s, AV_LOG_ERROR, "pyframe type not ready\n");
        return AVERROR(EINVAL);
    }
    Py_INCREF(&pymodule_PyFrameObjectType);
    if(PyModule_AddObject(s->module, "Frame", (PyObject *)&pymodule_PyFrameObjectType) < 0) {
        Py_DECREF(&pymodule_PyFrameObjectType);
        av_log(s, AV_LOG_ERROR, "add object failed\n");
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx) 
{
    PyModuleContext *s = ctx->priv;
    if(s->formats) {
        av_freep(&s->formats);
    }
    if(s->module) {
        Py_DECREF(s->module);
    }
    if(s->setup_args) {
        Py_DECREF(s->setup_args);
    }
    if(s->process_frame_args) {
        Py_DECREF(s->process_frame_args);
    }
}

static int query_formats(AVFilterContext *ctx)
{
    int ret;
    enum AVPixelFormat pixfmt;
    AVFilterFormats *fmts_list;
    PyObject *func, *formats;
    Py_ssize_t n_formats;
    PyModuleContext *s = ctx->priv;

    func = PyObject_GetAttrString(s->module, "query_formats");
    if(!func) {
        av_log(s, AV_LOG_WARNING, "query_formats() not found in %s \n", s->module_path);
        return AVERROR(EINVAL);
    }

    formats = PyObject_CallFunctionObjArgs(func, NULL);
    Py_DECREF(func);

    if(!formats || PyErr_Occurred()) {
        PyErr_Print();
        return AVERROR(EINVAL);
    }

    if(!PyList_Check(formats)) {
        av_log(s, AV_LOG_WARNING, "query_formats() should return a valid list %s \n", s->module_path);
        Py_DECREF(formats);
        return AVERROR(EINVAL);
    }

    if((n_formats = PyList_Size(formats)) <= 0) {
        av_log(s, AV_LOG_WARNING, "query_formats() return an empty list %s \n", s->module_path);
        Py_DECREF(formats);
        return AVERROR(EINVAL);
    }

    for(int k = 0; k < n_formats; k++) {
        PyObject *fmt_item = PyList_GetItem(formats, k);
        if(!PyUnicode_Check(fmt_item)) {
            continue;
        }

        PyObject *bytes = PyUnicode_AsEncodedString(
            fmt_item, "UTF-8", "strict");
        if (!bytes) {
            continue;
        }

        pixfmt = av_get_pix_fmt(PyBytes_AS_STRING(bytes));
        if(pixfmt == AV_PIX_FMT_NONE) {
            continue;
        }
        if(!av_dynarray2_add((void**)&s->formats, &s->nb_formats,
                            sizeof(*s->formats), (void *)&pixfmt)) {
            Py_DECREF(formats);
            return AVERROR(EINVAL);
        }
    }

    if(!s->nb_formats) {
        av_log(s, AV_LOG_WARNING, "query_formats() no available format %s \n", s->module_path);
        Py_DECREF(formats);
        return AVERROR(EINVAL);
    }

    pixfmt = AV_PIX_FMT_NONE;
    if(!av_dynarray2_add((void**)&s->formats, &s->nb_formats,
                            sizeof(*s->formats), (void *)&pixfmt)) {
        Py_DECREF(formats);
        return AVERROR(EINVAL);
    }

    fmts_list = ff_make_format_list(s->formats);
    if(!fmts_list) {
        Py_DECREF(formats);
        return AVERROR(ENOMEM);
    }

    if((ret = ff_set_common_formats(ctx, fmts_list)) < 0) {
        av_log(s, AV_LOG_WARNING, "ff_set_common_formats error!");
        Py_DECREF(formats);
        return ret;
    }
    Py_DECREF(formats);
    return 0;
}


#define OFFSET(x) offsetof(PyModuleContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption pymodule_options[] = {
    {"module",      "python file path",     OFFSET(module_path), AV_OPT_TYPE_STRING, { .str = NULL}, CHAR_MIN, CHAR_MAX, FLAGS },
    {"opts",        "module opts",          OFFSET(module_opts), AV_OPT_TYPE_STRING, { .str = NULL}, CHAR_MIN, CHAR_MAX, FLAGS },
    {NULL}
};

AVFILTER_DEFINE_CLASS(pymodule);

static const AVFilterPad avfilter_vf_pymodule_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    }, 
    {NULL}
};

static const AVFilterPad avfilter_vf_pymodule_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .request_frame = request_frame,
        .config_props = config_output,
    },
    {NULL}
};

AVFilter ff_vf_pymodule = {
    .name           = "pymodule",
    .description    = NULL_IF_CONFIG_SMALL("a python general filter"),
    .priv_size      = sizeof(PyModuleContext),
    .priv_class     = &pymodule_class,
    .init           = init,
    .uninit         = uninit,
    .query_formats  = query_formats,
    .inputs         = avfilter_vf_pymodule_inputs,
    .outputs        = avfilter_vf_pymodule_outputs,
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};