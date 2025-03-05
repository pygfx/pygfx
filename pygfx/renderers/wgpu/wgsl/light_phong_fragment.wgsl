var material: BlinnPhongMaterial;
material.diffuse_color = physical_albeido;
material.specular_color = srgb2physical(u_material.specular_color.rgb);
material.specular_shininess = u_material.shininess;
material.specular_strength = specular_strength;