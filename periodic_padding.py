import torch

def pos_int_to_base(x, b=2, extend_zero_to_length=0):
	if x == 0:
		return [0]
	out = []
	cur = x
	while cur > 0:
		out.append(int(cur % b))
		cur = int(cur / b)
	if extend_zero_to_length>0 and len(out) < extend_zero_to_length:
		out += [0]*(extend_zero_to_length - len(out))
	return out[::-1]

def compute_index_from_pad_region(region_code, dim_shape, dim_pad):
	original_lower = 0
	original_upper = 0
	out_lower = 0
	out_upper = 0
	if region_code == 1:
		out_upper = dim_pad[0]
		original_lower = dim_shape - dim_pad[0]
		original_upper = dim_shape
		return original_lower, original_upper, out_lower, out_upper

	if region_code == 2:
		out_lower = dim_shape + dim_pad[0]
		out_upper = dim_shape + dim_pad[0] + dim_pad[1]
		original_upper = dim_pad[1]
		return original_lower, original_upper, out_lower, out_upper

	original_upper = dim_shape
	out_lower = dim_pad[0]
	out_upper = dim_shape + dim_pad[0]
	return original_lower, original_upper, out_lower, out_upper

def periodic_padding_3d(x, pad):
	ndim = 3
	m = torch.nn.ConstantPad3d(pad,0)
	out = m(x)
	for i in range(1, 3**ndim):
		region_code = pos_int_to_base(i, 3, ndim)
		x_original_lower, x_original_upper, x_out_lower, x_out_upper =\
			compute_index_from_pad_region(region_code[0], int(x.shape[2]), pad[0:2])
		y_original_lower, y_original_upper, y_out_lower, y_out_upper =\
			compute_index_from_pad_region(region_code[1], int(x.shape[3]), pad[2:4])
		z_original_lower, z_original_upper, z_out_lower, z_out_upper =\
			compute_index_from_pad_region(region_code[2], int(x.shape[4]), pad[4:6])
		if x_out_lower != x_out_upper and\
			y_out_lower != y_out_upper and\
			z_out_lower != z_out_upper:

			out[:,:,x_out_lower: x_out_upper,
			y_out_lower: y_out_upper,
			z_out_lower: z_out_upper] =\
			x[:,:,x_original_lower: x_original_upper,
				y_original_lower: y_original_upper,
				z_original_lower: z_original_upper]
	return out

if __name__ == "__main__":
	x = torch.rand(3,4,2, 2, 2)
	out = periodic_padding_3d(x, pad=(1,1,1,1,1,1))
	print (out.shape)
