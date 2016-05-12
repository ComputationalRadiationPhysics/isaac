/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

namespace isaac
{

void mergeJSON(json_t* result,json_t* candidate)
{
	const char *c_key;
	const char *r_key;
	json_t *c_value;
	json_t *r_value;
	//metadata merge, old values stay, arrays are merged
	json_t* m_candidate = json_object_get(candidate, "metadata");
	json_t* m_result = json_object_get(result, "metadata");
	void *temp,*temp2;
	if (m_candidate && m_result)
	{
		json_object_foreach_safe( m_candidate, temp, c_key, c_value )
		{
			bool found_array = false;
			json_object_foreach_safe( m_result, temp2, r_key, r_value )
			{
				if (strcmp(r_key,c_key) == 0)
				{
					if (json_is_array(r_value) && json_is_array(c_value))
					{
						json_array_extend(r_value,c_value);
						found_array = true;
					}
					break;
				}
			}
			if (!found_array)
				json_object_set( m_result, c_key, c_value );
		}
	}            
	//general merge, new values stay
	json_object_foreach_safe( candidate, temp, c_key, c_value )
	{
		bool found_meta = false;
		json_object_foreach_safe( result, temp2, r_key, r_value )
		{
			if (strcmp(r_key,c_key) == 0 && strcmp(r_key,"metadata") == 0)
			{
				found_meta = true;
				break;
			}
		}
		if (!found_meta)
			json_object_set( result, c_key, c_value );
	}
}
void mulMatrixMatrix(IceTDouble* result,const IceTDouble* matrix1,const IceTDouble* matrix2)
{
	for (isaac_int x = 0; x < 4; x++)
		for (isaac_int y = 0; y < 4; y++)
			result[y+x*4] = matrix1[y+0*4] * matrix2[0+x*4]
						  + matrix1[y+1*4] * matrix2[1+x*4]
						  + matrix1[y+2*4] * matrix2[2+x*4]
						  + matrix1[y+3*4] * matrix2[3+x*4];
}
void mulMatrixVector(IceTDouble* result,const IceTDouble* matrix,const IceTDouble* vector)
{
	result[0] =  matrix[ 0] * vector[0] + matrix[ 4] * vector[1] +  matrix[ 8] * vector[2] + matrix[12] * vector[3];
	result[1] =  matrix[ 1] * vector[0] + matrix[ 5] * vector[1] +  matrix[ 9] * vector[2] + matrix[13] * vector[3];
	result[2] =  matrix[ 2] * vector[0] + matrix[ 6] * vector[1] +  matrix[10] * vector[2] + matrix[14] * vector[3];
	result[3] =  matrix[ 3] * vector[0] + matrix[ 7] * vector[1] +  matrix[11] * vector[2] + matrix[15] * vector[3];
}

void calcInverse(IceTDouble* inv,const IceTDouble* projection,const IceTDouble* modelview)
{
	IceTDouble m[16];
	mulMatrixMatrix(m,projection,modelview);
	inv[0] = m[5]  * m[10] * m[15] - 
			 m[5]  * m[11] * m[14] - 
			 m[9]  * m[6]  * m[15] + 
			 m[9]  * m[7]  * m[14] +
			 m[13] * m[6]  * m[11] - 
			 m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] + 
			  m[4]  * m[11] * m[14] + 
			  m[8]  * m[6]  * m[15] - 
			  m[8]  * m[7]  * m[14] - 
			  m[12] * m[6]  * m[11] + 
			  m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] - 
			 m[4]  * m[11] * m[13] - 
			 m[8]  * m[5] * m[15] + 
			 m[8]  * m[7] * m[13] + 
			 m[12] * m[5] * m[11] - 
			 m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] + 
			   m[4]  * m[10] * m[13] +
			   m[8]  * m[5] * m[14] - 
			   m[8]  * m[6] * m[13] - 
			   m[12] * m[5] * m[10] + 
			   m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] + 
			  m[1]  * m[11] * m[14] + 
			  m[9]  * m[2] * m[15] - 
			  m[9]  * m[3] * m[14] - 
			  m[13] * m[2] * m[11] + 
			  m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] - 
			 m[0]  * m[11] * m[14] - 
			 m[8]  * m[2] * m[15] + 
			 m[8]  * m[3] * m[14] + 
			 m[12] * m[2] * m[11] - 
			 m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] + 
			  m[0]  * m[11] * m[13] + 
			  m[8]  * m[1] * m[15] - 
			  m[8]  * m[3] * m[13] - 
			  m[12] * m[1] * m[11] + 
			  m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] - 
			  m[0]  * m[10] * m[13] - 
			  m[8]  * m[1] * m[14] + 
			  m[8]  * m[2] * m[13] + 
			  m[12] * m[1] * m[10] - 
			  m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] - 
			 m[1]  * m[7] * m[14] - 
			 m[5]  * m[2] * m[15] + 
			 m[5]  * m[3] * m[14] + 
			 m[13] * m[2] * m[7] - 
			 m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] + 
			  m[0]  * m[7] * m[14] + 
			  m[4]  * m[2] * m[15] - 
			  m[4]  * m[3] * m[14] - 
			  m[12] * m[2] * m[7] + 
			  m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] - 
			  m[0]  * m[7] * m[13] - 
			  m[4]  * m[1] * m[15] + 
			  m[4]  * m[3] * m[13] + 
			  m[12] * m[1] * m[7] - 
			  m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] + 
			   m[0]  * m[6] * m[13] + 
			   m[4]  * m[1] * m[14] - 
			   m[4]  * m[2] * m[13] - 
			   m[12] * m[1] * m[6] + 
			   m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] + 
			  m[1] * m[7] * m[10] + 
			  m[5] * m[2] * m[11] - 
			  m[5] * m[3] * m[10] - 
			  m[9] * m[2] * m[7] + 
			  m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] - 
			 m[0] * m[7] * m[10] - 
			 m[4] * m[2] * m[11] + 
			 m[4] * m[3] * m[10] + 
			 m[8] * m[2] * m[7] - 
			 m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] + 
			   m[0] * m[7] * m[9] + 
			   m[4] * m[1] * m[11] - 
			   m[4] * m[3] * m[9] - 
			   m[8] * m[1] * m[7] + 
			   m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] - 
			  m[0] * m[6] * m[9] - 
			  m[4] * m[1] * m[10] + 
			  m[4] * m[2] * m[9] + 
			  m[8] * m[1] * m[6] - 
			  m[8] * m[2] * m[5];

	IceTDouble det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return;
		
	det = 1.0 / det;

	for (isaac_int i = 0; i < 16; i++)
		inv[i] = inv[i] * det;
}

isaac_float4 getHSVA(isaac_float h, isaac_float s, isaac_float v, isaac_float a)
{
	isaac_int hi = isaac_int(floor(h / (M_PI/3)));
	isaac_float f = h / (M_PI/3) - isaac_float(hi);
	isaac_float p = v*(isaac_float(1)-s);
	isaac_float q = v*(isaac_float(1)-s*f);
	isaac_float t = v*(isaac_float(1)-s*(isaac_float(1)-f));
	isaac_float4 result = {0,0,0,a};
	switch (hi)
	{
		case 0: case 6:
			result.x = v;
			result.y = t;
			result.z = p;
			break;
		case 1:
			result.x = q;
			result.y = v;
			result.z = p;
			break;
		case 2:
			result.x = p;
			result.y = v;
			result.z = t;
			break;
		case 3:
			result.x = p;
			result.y = q;
			result.z = v;
			break;
		case 4:
			result.x = t;
			result.y = p;
			result.z = v;
			break;
		case 5:
			result.x = v;
			result.y = p;
			result.z = q;
			break;
	}
	return result;
}

void setFrustum(IceTDouble * const projection, const isaac_float left,const isaac_float right,const isaac_float bottom,const isaac_float top,const isaac_float znear,const isaac_float zfar )
{
	isaac_float  znear2 = znear * isaac_float(2);
	isaac_float  width = right - left;
	isaac_float  height = top - bottom;
	isaac_float  zRange = znear - zfar;
	projection[ 0] = znear2 / width;
	projection[ 1] = isaac_float( 0);
	projection[ 2] = isaac_float( 0);
	projection[ 3] = isaac_float( 0);
	projection[ 4] = isaac_float( 0);
	projection[ 5] = znear2 / height;
	projection[ 6] = isaac_float( 0);
	projection[ 7] = isaac_float( 0);
	projection[ 8] = ( right + left ) / width;
	projection[ 9] = ( top + bottom ) / height;
	projection[10] = ( zfar + znear) / zRange;
	projection[11] = isaac_float(-1);
	projection[12] = isaac_float( 0);
	projection[13] = isaac_float( 0);
	projection[14] = ( -znear2 * zfar ) / -zRange;
	projection[15] = isaac_float( 0);
}
void setPerspective(IceTDouble * const projection, const isaac_float fovyInDegrees,const isaac_float aspectRatio,const isaac_float znear,const isaac_float zfar )
{
	isaac_float ymax = znear * tan( fovyInDegrees * M_PI / isaac_float(360) );
	isaac_float xmax = ymax * aspectRatio;
	setFrustum(projection, -xmax, xmax, -ymax, ymax, znear, zfar );
}

void spSetPerspectiveStereoscopic( IceTDouble * const projection, const isaac_float fovyInDegrees,const isaac_float aspectRatio,const isaac_float znear,const isaac_float zfar,const isaac_float z0,const isaac_float distance )
{
	isaac_float t_z0 = znear * tan( fovyInDegrees * M_PI / isaac_float(360) );
	isaac_float xmin = -t_z0 + distance/2.0f*znear/z0;
	isaac_float xmax =  t_z0 + distance/2.0f*znear/z0;
	isaac_float ymin = -t_z0 / aspectRatio;
	isaac_float ymax =  t_z0 / aspectRatio;
	setFrustum(projection, xmin, xmax, ymin, ymax, znear, zfar );
	projection[12] += distance;
}


} //namespace isaac;
