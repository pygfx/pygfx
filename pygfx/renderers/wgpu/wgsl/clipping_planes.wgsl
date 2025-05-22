// ----- Clipping planes
$$ if n_clipping_planes
    $$ if clipping_mode == 'ANY'
        for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
            let plane = u_material.clipping_planes[i];
            if (dot(varyings.world_pos, plane.xyz) < plane.w) {
                discard;
            }
        }
    $$ else
        var clipped: bool = true;
        for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
            let plane = u_material.clipping_planes[i];
            if (dot(varyings.world_pos, plane.xyz) > plane.w) {
                clipped = false;
                break;  //at least one plane is outside, so we can keep the fragment
            }
        }
        if (clipped) {
            discard;
        }
    $$ endif
$$ endif