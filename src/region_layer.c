#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class_id >= 0){
            pred *= output[index + stride* class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride* class_id] = scale * (1 - output[index + stride* class_id]);

			class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride* class_id] = scale * (1 - output[index + stride* class_id]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class_id)?1 : 0) - output[index + stride*n]);
            if(n == class_id) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

// PREV:
// The last region layer in default has 13*13 region grids of one batch of image
// Each of region grids contains 5 anchor boxes with its data stride
// A data stride includes 4 coords, 1 weight and numbers(voc20 or coco80) of class id
//  => grid1[anchor1[x,y,w,h,obj,cls1~cls20],anchor2[x,y,w,h,obj,cls1~20],anchor3~5],grid2[anchor1~5],grid3~169
// CURR:
// The last region layer in default has 5 sequential data blocks for anchor boxes in one batch of image
// Each of data blocks represents a set of data includes 4 coords, 1 weight and numbers(voc20 or coco80) of class id
// A set contains 13*13 region grids of image
//  => anchor1[x[grid1~grid169]y[grid1~grid169]...cls20[grid1~grid169]],anchor2[x,y...cls20],anchor3~5

int get_box_index_old(layer l, int batch, int row, int col, int num, int offset)
{
	int batch_offset = batch * l.outputs;
	int grid = row * l.w + col;
	int anchor = grid * l.n + num;
	int anchor_offset = anchor * (l.coords + l.classes + 1);
	return batch_offset + anchor_offset + offset;
}

int get_box_index(layer l, int batch, int row, int col, int num, int offset)
{
	int batch_offset = batch * l.outputs;
	int anchor_offset = num * (l.coords + l.classes + 1) * (l.h * l.w);
	int data_offset = offset * (l.h * l.w);
	int grid_index = row * l.w + col;
	return batch_offset + anchor_offset + data_offset + grid_index;
}

box get_box(float *x, layer l, int batch, int row, int col, int num)
{
	int index = get_box_index(l, batch, row, col, num, 0);
	int stride = l.h * l.w;
	box b;
	b.x = ((float)col + x[index + 0 * stride]) / l.w;
	b.y = ((float)row + x[index + 1 * stride]) / l.h;
	b.w = expf(x[index + 2 * stride]) * l.biases[2 * num] / l.w;
	b.h = expf(x[index + 3 * stride]) * l.biases[2 * num + 1] / l.h;
	return b;
}

float set_delta_box(float *output, layer l, int batch, int row, int col, int num, box truth, float *delta, float scale)
{
	int index = get_box_index(l, batch, row, col, num, 0);
	int stride = l.h * l.w;
	box pred = get_box(output, l, batch, row, col, num);
	float iou = box_iou(pred, truth);

	float tx = (truth.x*l.w - col);
	float ty = (truth.y*l.h - row);
	float tw = logf(truth.w*l.w / l.biases[2 * num]);
	float th = logf(truth.h*l.h / l.biases[2 * num + 1]);

	delta[index + 0 * stride] = scale * (tx - output[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - output[index + 1 * stride]);
	delta[index + 2 * stride] = scale * (tw - output[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - output[index + 3 * stride]);
	return iou;
}

float set_delta_class(float *output, layer l, int batch, int row, int col, int num, int class_id, float *delta, float scale)
{
	float cls = 0;
	int index = get_box_index(l, batch, row, col, num, 5);
	int stride = l.h * l.w;

	if (l.softmax_tree) {
		float pred = 1;
		while (class_id >= 0) {
			pred *= output[index + stride * class_id];
			int g = l.softmax_tree->group[class_id];
			int offset = l.softmax_tree->group_offset[g];
			for (int i = 0; i < l.softmax_tree->group_size[g]; ++i) {
				delta[index + stride * (offset + i)] = scale * (0 - output[index + stride * (offset + i)]);
			}
			delta[index + stride * class_id] = scale * (1 - output[index + stride * class_id]);

			class_id = l.softmax_tree->parent[class_id];
		}
		cls = pred;
	}
	else {
		if (delta[index] && !l.softmax) {
			delta[index + stride * class_id] = scale * (1 - output[index + stride * class_id]);
			return cls;
		}
		for (int c = 0; c < l.classes; ++c) {
			if (c == class_id) {
				cls = output[index + stride * c];
				delta[index + stride * c] = scale * (1 - output[index + stride * c]);
			}
			else {
				delta[index + stride * c] = scale * (0 - output[index + stride * c]);
			}
		}
	}
	return cls;
}

void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < l.max_boxes; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class_id = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class_id, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class_id, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
					//int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    //box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
					box pred = get_box(l.output, l, b, j, i, n);
					float best_iou = 0;
					int best_class_id = -1;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
							best_class_id = net.truth[t*(l.coords + 1) + b * l.truths + 4];
                            best_iou = iou;
                        }
                    }
					int obj_index = get_box_index(l, b, j, i, n, 4);
                    //int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
					l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
					if (l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }

                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        //delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
						set_delta_box(l.output, l, b, j, i, n, truth, l.delta, 0.01);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for(n = 0; n < l.n; ++n){
				//int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                //box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
				box pred = get_box(l.output, l, b, j, i, n);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

			//int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            //float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
			float iou = set_delta_box(l.output, l, b, j, i, best_n, truth, l.delta, l.coord_scale * (2 - truth.w * truth.h));
			if(l.coords > 4){
				int mask_index = get_box_index(l, b, j, i, best_n, 4);
				//int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

			int obj_index = get_box_index(l, b, j, i, best_n, 4);
			//int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
			l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
				l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background){
				l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }

            int class_id = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class_id = l.map[class_id];
			//int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            //delta_region_class(l.output, l.delta, class_index, class_id, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
			float cls = set_delta_class(l.output, l, b, j, i, best_n, class_id, l.delta, l.class_scale);
			avg_cat += cls;
			++count;
            ++class_count;
        }

#ifndef GPU
		for (n = 0; n < l.n; n++) {
			int index = get_box_index(l, b, 0, 0, n, 0);
			gradient_array(l.output + index, 2 * l.w*l.h, LOGISTIC, l.delta);
			index = get_box_index(l, b, 0, 0, n, 4);
			if (!l.background) gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta);
		}
		float cost_ratio = 1;
		*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2) * cost_ratio;
#endif
    }
    //printf("\n");
    
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_box(predictions, l, 0, row, col, n);
			//dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            } else {
                if(dets[index].objectness){
                    for(j = 0; j < l.classes; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = get_box_index(l, b, 0, 0, n, 0);
			//int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
			if (l.coords > 4) {
				index = get_box_index(l, b, 0, 0, n, 4);
				//index = entry_index(l, b, n*l.w*l.h, 4);
				activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
			}
			index = get_box_index(l, b, 0, 0, n, 4);
			//index = entry_index(l, b, n*l.w*l.h, l.coords);
			if (!l.background) activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
			index = get_box_index(l, b, 0, 0, n, 5);
			//index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
			if (!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
		}
	}

	if (l.softmax_tree) {
		int index = get_box_index(l, 0, 0, 0, 0, 5);
		//int index = entry_index(l, 0, 0, l.coords + 1);
		softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs / l.n, 1, l.output_gpu + index, *l.softmax_tree);
	}
	else if (l.softmax) {
		int index = get_box_index(l, 0, 0, 0, 0, (l.background) ? 4 : 5);
		//int index = entry_index(l, 0, 0, l.coords + !l.background);
		//printf("%d\n", index);
		//softmax_gpu(l.output_gpu + 5, l.classes, l.batch, l.inputs, l.h*l.w*l.n, l.coords + l.classes + 1, 1, 1, l.output_gpu + 5);
		softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
	}

    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = get_box_index(l, b, 0, 0, n, 0);
			//int index = entry_index(l, b, n*l.w*l.h, 0);
			gradient_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC, l.delta_gpu + index);
			if (l.coords > 4) {
				index = get_box_index(l, b, 0, 0, n, 4);
				//index = entry_index(l, b, n*l.w*l.h, 4);
				gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
			}
			index = get_box_index(l, b, 0, 0, n, 4);
			//index = entry_index(l, b, n*l.w*l.h, l.coords);
			if (!l.background) gradient_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC, l.delta_gpu + index);
		}
	}
	cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	float cost_ratio = 1;
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2) * cost_ratio;
}

void backward_region_layer_gpu(const layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}

