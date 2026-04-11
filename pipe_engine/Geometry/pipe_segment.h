#pragma once
#include "vec3.h"
#include <vector>
#include <string>
#include <cmath>

// ============================================================================
// PipeSegment: base abstraction for a pipe primitive.
//
// Every segment knows how to sample its centre-line at parameter t in [0,1]
// and provide its local tangent.  The mesh generator sweeps a circular
// cross-section along that centre-line.
// ============================================================================

enum class SegmentType { Straight, Bend };

struct PipeSegment {
    SegmentType type;
    float innerRadius = 0.05f;   // metres
    float outerRadius = 0.06f;   // metres (innerRadius + wall thickness)
    std::string label;           // optional ID for blueprint references

    virtual ~PipeSegment() = default;

    // Parameter t in [0,1].  Returns centre-line position.
    virtual Vec3 centreLine(float t)  const = 0;
    // Tangent (unit-length forward direction) at parameter t.
    virtual Vec3 tangent(float t)     const = 0;
    // Total arc length of the centre-line.
    virtual float arcLength()         const = 0;
    // Start / end points for easy connectivity checks.
    Vec3 startPoint() const { return centreLine(0.f); }
    Vec3 endPoint()   const { return centreLine(1.f); }
};

// ============================================================================
// StraightSegment
// ============================================================================

struct StraightSegment : PipeSegment {
    Vec3 origin;      // start of the segment
    Vec3 direction;   // unit direction
    float length;     // metres

    StraightSegment() { type = SegmentType::Straight; length = 1.f; direction = {0,0,1}; }

    StraightSegment(const Vec3& from, const Vec3& to,
                    float innerR = 0.05f, float outerR = 0.06f)
    {
        type = SegmentType::Straight;
        origin = from;
        Vec3 d = to - from;
        length = d.length();
        direction = (length > 1e-12f) ? d / length : Vec3{0,0,1};
        innerRadius = innerR;
        outerRadius = outerR;
    }

    Vec3  centreLine(float t) const override { return origin + direction * (t * length); }
    Vec3  tangent(float)      const override { return direction; }
    float arcLength()         const override { return length; }
};

// ============================================================================
// BendSegment — a circular arc in 3D
//
// Defined by:
//   centre     – centre of curvature
//   startDir   – unit vector from centre to the start of the arc
//   normal     – axis of rotation (cross product of start and end radii)
//   bendRadius – distance from centre to the centre-line
//   angleRad   – sweep angle in radians (positive = CCW around normal)
// ============================================================================

struct BendSegment : PipeSegment {
    Vec3  centre;
    Vec3  startDir;    // unit: centre → start of arc
    Vec3  normal;      // rotation axis (unit)
    float bendRadius;  // radius of curvature
    float angleRad;    // sweep angle

    BendSegment() {
        type = SegmentType::Bend;
        bendRadius = 0.2f;
        angleRad   = float(M_PI) * 0.5f;
    }

    // Construct a 90-degree elbow that smoothly connects two directions.
    // inDir  : unit direction entering the bend
    // outDir : unit direction leaving the bend
    // startPt: where the straight section ends (= where the bend begins)
    static BendSegment makeElbow(const Vec3& startPt,
                                 const Vec3& inDir,
                                 const Vec3& outDir,
                                 float bendR,
                                 float innerR = 0.05f,
                                 float outerR = 0.06f)
    {
        BendSegment b;
        b.innerRadius = innerR;
        b.outerRadius = outerR;
        b.bendRadius  = bendR;

        // The rotation axis is perpendicular to both directions
        Vec3 axis = inDir.cross(outDir).normalized();
        if (axis.lengthSq() < 1e-8f) {
            // directions are (anti-)parallel — pick arbitrary axis
            Vec3 up = (std::abs(inDir.y) < 0.99f) ? Vec3{0,1,0} : Vec3{1,0,0};
            axis = inDir.cross(up).normalized();
        }
        b.normal = axis;

        // Sweep angle: the pipe must turn from inDir to outDir.
        // The arc sweeps by the supplement of the angle between the directions.
        float cosA = std::clamp(inDir.dot(outDir), -1.f, 1.f);
        float angleBetween = std::acos(cosA);
        b.angleRad = float(M_PI) - angleBetween;
        // Special case: anti-parallel directions → full 180° U-bend
        if (angleBetween > float(M_PI) - 1e-4f) {
            b.angleRad = float(M_PI);
        }

        // Centre of curvature: offset from startPt perpendicular to inDir
        // in the plane defined by inDir and outDir
        Vec3 towardsCentre = axis.cross(inDir).normalized();
        b.centre   = startPt + towardsCentre * bendR;
        b.startDir = (startPt - b.centre).normalized();

        return b;
    }

    Vec3 centreLine(float t) const override {
        float theta = t * angleRad;
        // Rodrigues' rotation of startDir around normal by theta
        Vec3 v = startDir;
        float cosT = std::cos(theta), sinT = std::sin(theta);
        Vec3 rotated = v * cosT
                     + normal.cross(v) * sinT
                     + normal * (normal.dot(v) * (1.f - cosT));
        return centre + rotated * bendRadius;
    }

    Vec3 tangent(float t) const override {
        float theta = t * angleRad;
        Vec3 v = startDir;
        float cosT = std::cos(theta), sinT = std::sin(theta);
        Vec3 rotated = v * cosT
                     + normal.cross(v) * sinT
                     + normal * (normal.dot(v) * (1.f - cosT));
        // tangent = d/dθ of (centre + R * rotated) / |...|  =  normal × rotated
        Vec3 tang = normal.cross(rotated).normalized();
        return tang;
    }

    float arcLength() const override {
        return bendRadius * std::abs(angleRad);
    }
};
