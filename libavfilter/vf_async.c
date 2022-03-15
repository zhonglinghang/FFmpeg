#include <pthread.h>
#include "internal.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "avfilter.h"
#include "list.h"

#define MAX_INQ_SIZE 32
#define MAX_OUTQ_SIZE 32

#define MAX_ARGS_LEN 1024
#define MAX_THREADS_NUM 16

typedef struct
{
    /* data */
    AVFilterGraphAsync fctx;
    pthread_t tid;
    int index;
    void *this;
} SyncThread;

typedef struct {
    AVFilterLink *inlink;
    AVFilterLink *outlink;
    pthread_mutex_t imutex;
    pthread_mutex_t omutex;
    pthread_mutex_t fctx_mutex;
    list_t *ifq;
    char args[MAX_ARGS_LEN];
    int abort;
    int threads;
    int input_frame_seq;
    SyncThread tinfo[MAX_THREADS_NUM];
    AVFrame *ofq[MAX_OUTQ_SIZE];
    int rindex;
    int cache_size;
    int thread_pool_num;
} AsyncContext;

static void* sync_thread_func(void* args) 
{
    SyncThread *tinfo = args;
    AsyncContext *async = (AsyncContext *)(tinfo->this);
    int index = tinfo->index;
    list_node_t *node = NULL;
    AVFrame *frame = NULL;
    uint64_t findex = 0;
    int ret = 0;
    int iq_size = 0;
    int full_log = 0;
    while(!async->abort) {
        AVFrame *out = NULL;
        AVFilterGraphAsync *fctx = NULL;
        pthread_mutex_lock(&async->imutex);
        iq_size = list_size(async->ifq);
        if(!iq_size) {
            pthread_mutex_unlock(&async->imutex);
            av_usleep(5000);
            continue;
        } else if (iq_size == MAX_INQ_SIZE) {
            if (full_log == 0) {
                full_log = 1;
                av_log(NULL, AV_LOG_TRACE, "async [%s] filter in queue full, scale input size:%d.\n", async->args, iq_size);
            }
        } else if(full_log && iq_size < 10) {
            full_log = 0;
        }
        node = list_lpop(async->ifq);
        if(!node) {
            pthread_mutex_unlock(&async->imutex);
            continue;
        }
        pthread_mutex_unlock(&async->imutex);
        frame = node->val;
        LIST_FREE(node);
        fctx = &async->tinfo[index].fctx;
        avfilter_graph_async_init_fg(fctx, async->args, frame->width, frame->height, frame->format, frame->sample_aspect_ratio, async->outlink->format, async->thread_pool_num);
        if(!fctx->filter_graph) {
            av_frame_free(&frame);
            av_log(NULL, AV_LOG_ERROR, "async [%s] filter graph not exist\n", async->args);
            continue;
        }
        ret = avfilter_graph_async_filter_frame(fctx, frame, &out);
        if (ret < 0) {
            if(ret == AVERROR(EAGAIN)) {
                continue;
            }
            av_log(NULL, AV_LOG_ERROR, "async [%s] filter graph filter frame failed: %s.\n", async->args, av_err2str(ret));
            break;
        }
        if(out == NULL) {
            continue;
        }

        while(!async->abort) {
            pthread_mutex_lock(&async->omutex);
            findex = out->decode_error_flags % MAX_OUTQ_SIZE;
            if(async->ofq[findex]) {
                pthread_mutex_unlock(&async->omutex);
                av_usleep(2 * 1000);
                continue;
            }
            async->ofq[findex] = out;
            pthread_mutex_unlock(&async->omutex);
            break;
        }
    }

    pthread_mutex_lock(&async->fctx_mutex);
    avfilter_graph_async_uninit_fg(&async->tinfo[index].fctx);
    pthread_mutex_unlock(&async->fctx_mutex);
    return NULL;
}

static int filter_frame(AVFilterLink *link, AVFrame *in) 
{
    AsyncContext *async = link->dst->priv;
    AVFilterLink *outlink = link->dst->outputs[0];
    AVFrame *out = NULL;
    int ret = 0, size = 0;
    list_node_t *node = NULL;
    int empty = 0;
    pthread_mutex_lock(&async->omutex);
    if(async->rindex > MAX_OUTQ_SIZE) {
        av_log(NULL, AV_LOG_ERROR, "async rindex error");
        empty = 1;
        pthread_mutex_unlock(&async->omutex);
        goto end;
    }
    out = async->ofq[async->rindex];
    if(!out) {
        static int s_failed_frame_ecc = 0;
        pthread_mutex_unlock(&async->omutex);
        empty = 1;
        if((s_failed_frame_ecc++) % 100 == 0) {
            av_log(NULL, AV_LOG_ERROR, "async [%s] filter got frame failed (err=%d)", async->args, s_failed_frame_ecc);
        }
        goto end;
    }
    async->ofq[async->rindex] = NULL;
    async->rindex = (async->rindex >= (MAX_OUTQ_SIZE - 1)) ? 0 : (async->rindex + 1);
    async->cache_size --;
    pthread_mutex_unlock(&async->omutex);

    if (out->width != outlink->w || out->height != outlink->h || out->format != outlink->format
        || out->sample_aspect_ratio.num != outlink->sample_aspect_ratio.num
        || out->sample_aspect_ratio.den != outlink->sample_aspect_ratio.den) {
        outlink->h = out->height;
        outlink->w = out->width;
        outlink->format = out->format;
        outlink->sample_aspect_ratio = out->sample_aspect_ratio;
    }
end:
    while(!async->abort) {
        pthread_mutex_lock(&async->imutex);
        if(list_is_full(async->ifq)) {
            pthread_mutex_unlock(&async->imutex);
            av_usleep(2 * 1000);
            continue;
        }
        in->decode_error_flags = async->input_frame_seq++;
        if(async->input_frame_seq == MAX_OUTQ_SIZE) {
            async->input_frame_seq = 0;
        }
        list_rpush(async->ifq, list_node_new(in));
        async->cache_size++;
        pthread_mutex_unlock(&async->imutex);
        break;
    }

    if(empty) {
        return 0;
    }
    return ff_filter_frame(outlink, out);
}

static int query_formats(AVFilterContext *ctx) 
{
    AsyncContext *async = ctx->priv;
    char fname[MAX_ARGS_LEN] = {0};
    AVFilter *f = NULL;
    sscanf(async->args, "%[^=]", fname);
    av_log(NULL, AV_LOG_ERROR, "async args[%s] filter[%s]\n", async->args, fname);
    f = avfilter_get_by_name(fname);
    if (!f) {
        return -1;
    }
    return f->query_formats(ctx);
}

static void frame_queue_list_node_free_func(void *node_data) 
{
    AVFrame *frame = (AVFrame *) node_data;
    if (frame) {
        av_frame_free(&frame);
    }
}

static av_cold int init(AVFilterContext *ctx)
{
    AsyncContext *async = ctx->priv;

    pthread_mutex_init(&async->imutex, NULL);
    pthread_mutex_init(&async->omutex, NULL);
    pthread_mutex_init(&async->fctx_mutex, NULL);

    async->ifq = list_new(MAX_INQ_SIZE);
    async->ifq->free = frame_queue_list_node_free_func;

    async->abort = 0;
    async->input_frame_seq = 0;
    async->rindex = 0;
    pthread_mutex_lock(&async->omutex);
    memset(async->ofq, 0x00, sizeof(async->ofq));
    pthread_mutex_unlock(&async->omutex);
    memset(async->tinfo, 0x00, sizeof(async->tinfo));
    return 0;
}

static av_cold int uninit(AVFilterContext *ctx) {
    AsyncContext *async = ctx->priv;
    async->abort = 1;
    for(int i = 0; i < async->threads; i++){
        pthread_join(async->tinfo[i].tid, NULL);
    }

    pthread_mutex_lock(&async->imutex);
    list_remove_until(async->ifq, NULL);
    pthread_mutex_unlock(&async->imutex);

    pthread_mutex_lock(&async->omutex);
    for(int i = 0; i < MAX_OUTQ_SIZE; i++) {
        if(async->ofq[i]) {
            av_frame_free(&async->ofq[i]);
        }
    }
    memset(async->ofq, 0x00, sizeof(async->ofq));
    pthread_mutex_unlock(&async->omutex);

    pthread_mutex_destroy(&async->imutex);
    pthread_mutex_destroy(&async->omutex);
    pthread_mutex_destroy(&async->fctx_mutex);

    async->input_frame_seq = 0;
    async->rindex = 0;
    memset(async->tinfo, 0x00, sizeof(async->tinfo));
    
    return 0;
}

static int filter_process_options(AVFilterContext *ctx, AVDictionary **options, const char *args)
{
    AsyncContext *async = NULL;
    char *s = NULL, *e = NULL;
    int len = 0;
    char threads[128];

    if (args == NULL || strlen(args) < 1 || ctx == NULL) {
        return -1;
    }
    if(strlen(args) >= MAX_ARGS_LEN - 1) {
        av_log(NULL, AV_LOG_ERROR, "args too long: %d > %d: %s", strlen(args), MAX_ARGS_LEN - 1, args);
        return -1;
    }
    async = ctx->priv;
    if(!async) {
        return -1;
    }
    async->threads = 1;
    sscanf(args, "%*[^{]{%[^}]", async->args);
    s = strstr(args, "threads=");
    if(s) {
        e = strstr(s, ":");
        if(e) {
            len = e - s - 8;
        } else {
            len = strlen(s) - 8;
        }
        snprintf(threads, len + 1, "%s", s + 8);
        async->threads = atoi(threads);
    } else {
        async->threads = 1;
    }

    async->threads = async->threads > 16 ? 16 : async->threads;

    s = strstr(args, "tpn=");
    if (s) {
        e = strchr(s, ":");
        if (e) {
            len = e - s - strlen("tpn=");
        } else {
            len = strlen(s) - strlen("tpn=");
        }
        snprintf(threads, len + 1, "%s", s + strlen("tpn="));
        async->thread_pool_num = atoi(threads);
    } else {
        async->thread_pool_num = 1;
    }
    return 1;
}

static int config_input(AVFilterLink *inlink) 
{
    AsyncContext *async = inlink->dst->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    if(!inlink || !async) {
        return -1;
    }
    async->inlink = inlink;
    async->outlink = outlink;
    return 0;
}

static int config_output(AVFilterLink *outlink) 
{
    AsyncContext *async = outlink->src->priv;
    AVFilterLink *inlink = outlink->src->inputs[0];

    for(int i = 0; i < async->threads; i++) {
        AVFilterGraphAsync *fctx = NULL;
        pthread_mutex_lock(&async->fctx_mutex);
        fctx = &async->tinfo[i].fctx;
        if(i == 0) {
            avfilter_graph_async_init_fg(fctx, async->args, inlink->w, inlink->h, inlink->format, inlink->sample_aspect_ratio, outlink->format, async->thread_pool_num);
        }
        pthread_mutex_unlock(&async->fctx_mutex);

        async->tinfo[i].this = async;
        async->tinfo[i].index = i;
        if (pthread_create(&async->tinfo[i].tid, NULL, sync_thread_func, &async->tinfo[i]) != 0) {
            return -1;
        }
    }

    if(async->tinfo[0].fctx.filter_graph && async->tinfo[0].fctx.filter_graph->sink_links[0]) {
        outlink->w = async->tinfo[0].fctx.filter_graph->sink_links[0]->w;
        outlink->h = async->tinfo[0].fctx.filter_graph->sink_links[0]->h;
        outlink->format = async->tinfo[0].fctx.filter_graph->sink_links[0]->format;
        outlink->sample_aspect_ratio = async->tinfo[0].fctx.filter_graph->sink_links[0]->sample_aspect_ratio;
    } else {
        av_log(NULL, AV_LOG_ERROR, "async config output failed. \n");
        return -1;
    }
    return 0;
}

#define OFFSET(x) offsetof(AsyncContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption async_options[] = {
    {NULL}
};

AVFILTER_DEFINE_CLASS(async);

static const AVFilterPad avfilter_vf_async_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    }, 
    {NULL}
};

static const AVFilterPad avfilter_vf_async_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_output,
    },
    {NULL}
};

AVFilter ff_vf_async = {
    .name = "async",
    .description = NULL_IF_CONFIG_SMALL("async filter frame with thread."),
    .priv_size = sizeof(AsyncContext),
    .priv_class = &async_class,
    .query_formats = query_formats,
    .init = init,
    .uninit = uninit,
    .inputs = avfilter_vf_async_inputs,
    .outputs = avfilter_vf_async_outputs,
    .flags = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    .process_options = filter_process_options,
};